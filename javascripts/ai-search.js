
// 100% 在瀏覽器端執行的語意搜尋：完全不需要 app.py / localhost:8000 這種
// 本機後端伺服器，模型 (transformers.js + ONNX) 和向量資料庫都是直接從
// CDN／同一個靜態網站下載到使用者的瀏覽器裡執行，所以整個網站可以單純當作
// 靜態檔案放在 GitHub Pages 上運作。
import {
  pipeline,
  env,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.2";

// 不要嘗試去讀取本機檔案系統上的模型，一律從 Hugging Face Hub 的 CDN 下載，
// 這樣才符合「開啟網頁就自動下載模型」的需求。
env.allowLocalModels = false;

const MODEL_ID = "Xenova/all-MiniLM-L6-v2"; // 對應 model.txt 選定的輕量模型
const META_URL = "docs-meta.json"; // build_client_embeddings.py 產生
const EMBEDDINGS_URL = "docs-embeddings.bin"; // 同上，float32 二進位向量
const DEBOUNCE_MS = 300;
// MiniLM 的餘弦相似度分佈跟 e5 不同、也沒有那麼可靠：實測完全無關的查詢
// 最高分也常落在 0.25~0.3，真正相關的英文查詢多半在 0.45 以上，中文查詢因為
// 這顆模型對中文語意的理解本來就比較弱 (見 model.txt)，分數會再更低。
// 這個門檻只用來擋「沒有命中任何關鍵字」的片段，命中關鍵字的片段一律保留
// (見 keywordBoost / matched)，所以就算門檻抓得寬鬆一點，也不會被雜訊灌爆。
const SIM_THRESHOLD = 0.3;
// 不要固定只顯示前 5 筆：片段現在切得很細 (一個項目/一段話就是一筆)，
// 有時候真正相關的結果只有 1、2 筆，硬湊滿 5 筆會塞進不太相關的內容；
// 有時候同一個章節底下有 8、9 個子項目都跟查詢同樣相關，硬砍成 5 筆
// 又會漏掉本來該顯示的結果。改成看分數：只要分數達到最高分的這個比例
// 以上，就一起顯示，讓筆數自然跟著「這次查詢到底有多少相關結果」走。
const RESULT_RELATIVE_CUTOFF = 0.7;
// 極端情況 (例如查詢字詞多、命中太多片段) 的保底上限，避免整頁被灌爆。
const RESULT_MAX_COUNT = 15;

const state = {
  extractor: null, // transformers.js 的 feature-extraction pipeline
  documents: null, // [{title, url, text}, ...]
  embeddings: null, // Float32Array，row-major，每列 dim 長
  dim: 0,
  ready: false,
  modelProgress: 0,
  dataProgress: 0,
};

let debounceTimer = null;
let requestSeq = 0;

function getInput() {
  return document.querySelector('[data-md-component="search-query"]');
}

function getResultBox() {
  const box = document.querySelector('[data-md-component="search-result"]');
  if (!box) return null;
  return {
    meta: box.querySelector(".md-search-result__meta"),
    list: box.querySelector(".md-search-result__list"),
  };
}

function setMeta(text) {
  const box = getResultBox();
  if (box && box.meta) box.meta.textContent = text;
}

function setList(html) {
  const box = getResultBox();
  if (box && box.list) box.list.innerHTML = html;
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function showLoadingStatus() {
  const pct = Math.round((state.modelProgress * 0.7 + state.dataProgress * 0.3) * 100);
  setMeta(`AI 搜尋引擎啟動中，正在下載模型與向量資料庫… ${pct}%`);
  setList("");
}

// 載入 all-MiniLM-L6-v2 的 ONNX 權重 (量化版，約 0.7MB)，下載進度透過
// progress_callback 回報，讓使用者知道還要等多久，而不是整頁沒反應。
async function loadModel() {
  state.extractor = await pipeline("feature-extraction", MODEL_ID, {
    dtype: "q8",
    progress_callback: (data) => {
      if (data.status === "progress" && data.total) {
        state.modelProgress = data.loaded / data.total;
        if (!state.ready) showLoadingStatus();
      }
    },
  });
  state.modelProgress = 1;
}

// 下載預先算好的文件向量資料庫 (docs-meta.json + docs-embeddings.bin)，
// 這兩個檔案由 build_client_embeddings.py 離線產生，跟這個網站一起發布，
// 瀏覽器只需要 fetch 下來，不用自己重新對整份文件庫做 embedding。
async function loadDocsDatabase() {
  const [metaRes, binRes] = await Promise.all([
    fetch(META_URL),
    fetch(EMBEDDINGS_URL),
  ]);

  if (!metaRes.ok || !binRes.ok) {
    throw new Error("無法下載向量資料庫檔案");
  }

  const meta = await metaRes.json();
  const buffer = await binRes.arrayBuffer();

  state.documents = meta.documents;
  state.dim = meta.dim;
  state.embeddings = new Float32Array(buffer);
  state.dataProgress = 1;
}

// 計算查詢向量跟資料庫中每一列文件向量的餘弦相似度。
// 因為 build_client_embeddings.py 存檔前就已經對每個向量做過 L2 normalize，
// 這裡的向量長度都是 1，所以「餘弦相似度」直接用內積 (dot product) 算就好，
// 不用再各自除以向量長度。
function cosineSimilarityAll(queryVec) {
  const { embeddings, dim, documents } = state;
  const n = documents.length;
  const scores = new Float32Array(n);

  for (let i = 0; i < n; i++) {
    const offset = i * dim;
    let dot = 0;
    for (let d = 0; d < dim; d++) {
      dot += embeddings[offset + d] * queryVec[d];
    }
    scores[i] = dot;
  }

  return scores;
}

// 跟 app.py 的 keyword_boost 邏輯類似，但這裡的判斷比重更高：
// all-MiniLM-L6-v2 是小模型，對「一個產品型號單字」這種很短的查詢，
// 語意分數常常跟完全不相關的雜訊字串重疊在同一個區間 (實測 0.15~0.3 都有可能)，
// 沒辦法只靠一個語意分數門檻可靠篩選。所以只要片段裡真的完整出現查詢字串
// (或每個查詢字詞)，就視為「命中關鍵字」，不管語意分數多低都不該被濾掉，
// 語意分數在這種情況下只用來排序，不用來把關。
//
// idfWeights: 像「core2 led on」這種「產品型號 + 通用字詞」的查詢，若每個
// 命中字詞都給同樣的加分，"led"/"on" 這種幾乎每一頁都有的字會稀釋掉
// "core2" 這種真正能鎖定產品的關鍵字，導致排名前面全是別的產品、但同樣
// 提到 LED 的片段。所以命中字詞的加分要依照它在整個資料庫裡多罕見來加權
// (類似 IDF)：越少片段出現的字詞，加分越高；title 命中的加分也比 text 命中高，
// 因為產品型號通常只會出現在標題。
function keywordBoost(title, text, queryLower, queryWords, idfWeights) {
  const titleLower = title.toLowerCase();
  const textLower = text.toLowerCase();
  let boost = 0;
  let matched = false;
  if (queryLower && (titleLower + " " + textLower).includes(queryLower)) {
    boost += 0.15;
    matched = true;
  }
  if (queryWords.length > 1) {
    let hitCount = 0;
    for (const w of queryWords) {
      const idf = idfWeights[w] || 1;
      if (titleLower.includes(w)) {
        boost += 0.06 * idf;
        hitCount++;
      } else if (textLower.includes(w)) {
        boost += 0.015 * idf;
        hitCount++;
      }
    }
    if (hitCount === queryWords.length) matched = true;
  }
  return { boost, matched };
}

// 用查詢字詞在整個文件庫裡出現的片段數估計「這個字詞有多罕見」，愈少片段
// 提到就給愈高權重 (類似 IDF)。是在目前已經下載到瀏覽器記憶體裡的
// state.documents 上即時算，不用額外下載資料，1700 多筆片段掃幾個字詞
// 花不到幾毫秒。
function computeIdfWeights(queryWords) {
  const docs = state.documents;
  const n = docs.length;
  const weights = {};
  for (const w of queryWords) {
    let df = 0;
    for (const doc of docs) {
      if ((doc.title + " " + doc.text).toLowerCase().includes(w)) df++;
    }
    weights[w] = Math.log((n + 1) / (df + 1)) + 0.5;
  }
  return weights;
}

// 兩個字串之間的編輯距離 (Levenshtein distance)，字愈短計算量愈小，
// 用在單一詞彙比對上 (十幾個字元內) 幾乎不花時間。
function levenshteinDistance(a, b) {
  const m = a.length;
  const n = b.length;
  if (m === 0) return n;
  if (n === 0) return m;
  const dp = new Array(n + 1);
  for (let j = 0; j <= n; j++) dp[j] = j;
  for (let i = 1; i <= m; i++) {
    let prevDiag = dp[0];
    dp[0] = i;
    for (let j = 1; j <= n; j++) {
      const temp = dp[j];
      dp[j] = a[i - 1] === b[j - 1]
        ? prevDiag
        : 1 + Math.min(prevDiag, dp[j], dp[j - 1]);
      prevDiag = temp;
    }
  }
  return dp[n];
}

// 0~1 的相似度，1 代表完全一樣
function wordSimilarity(a, b) {
  if (!a.length || !b.length) return 0;
  return 1 - levenshteinDistance(a, b) / Math.max(a.length, b.length);
}

const HIGHLIGHT_SIMILARITY = 0.85;

// 把片段裡「跟查詢字詞相近」的字都用 <mark> 包起來，讓使用者一眼就能看到
// 「為什麼是這個結果」。不是要求整個字完全等於查詢字詞才醒目提示 —
// 那樣查 "channel" 會因為文件裡寫的是 "channels" (多了一個 s) 就完全比對不到，
// 一個都標不出來。改成逐字比對：只要文件裡的字包含查詢字詞 (或反過來)，
// 或是兩者的編輯距離相似度達到 85% 以上 (能抓到單複數、動詞變化、小拼字差異
// 這類「很像但不完全一樣」的情況)，就當作命中。不特別標數字，避免跟查詢
// 無關的數字也被醒目提示，反而分散注意力。
// 一定要先 escapeHtml 再插入 <mark>，避免片段內容裡本來就有的 < > & 被誤判成標籤。
function highlightSnippet(text, queryWords) {
  const escaped = escapeHtml(text);
  const words = queryWords.filter(Boolean);
  if (!words.length) return escaped;

  return escaped.replace(/[A-Za-z0-9']+/g, (token) => {
    const tokenLower = token.toLowerCase();
    const isMatch = words.some((w) => {
      if (tokenLower.includes(w) || w.includes(tokenLower)) return true;
      return wordSimilarity(tokenLower, w) >= HIGHLIGHT_SIMILARITY;
    });
    return isMatch ? `<mark>${token}</mark>` : token;
  });
}

function renderResults(results, queryWords) {
  if (!results.length) {
    setMeta("沒有找到符合的文件");
    setList("");
    return;
  }
  setMeta(results.length + " 筆結果");
  const html = results
    .map(
      (r) =>
        '<li class="md-search-result__item">' +
        '<a href="' + escapeHtml(r.url) + '" class="md-search-result__link">' +
        '<article class="md-search-result__article md-search-result__article--document">' +
        "<h1>" + highlightSnippet(r.title, queryWords) + "</h1>" +
        "<p>" + highlightSnippet(r.text, queryWords) + "</p>" +
        "</article>" +
        "</a>" +
        "</li>"
    )
    .join("");
  setList(html);
}

async function runSearch(query) {
  if (!state.ready) {
    showLoadingStatus();
    return;
  }

  setMeta("搜尋中…");
  const seq = ++requestSeq;

  // MiniLM 不像 E5 需要 "query: " / "passage: " 前綴，直接編碼原始字句即可。
  const output = await state.extractor(query, { pooling: "mean", normalize: true });
  if (seq !== requestSeq) return; // 有更新的查詢已送出，捨棄過期結果
  const queryVec = output.data;

  const scores = cosineSimilarityAll(queryVec);

  const queryLower = query.trim().toLowerCase();
  const queryWords = queryLower.split(/\s+/).filter(Boolean);
  const idfWeights = queryWords.length > 1 ? computeIdfWeights(queryWords) : {};

  const scored = state.documents.map((doc, i) => {
    const { boost, matched } = keywordBoost(doc.title, doc.text, queryLower, queryWords, idfWeights);
    return { doc, baseScore: scores[i], adjustedScore: scores[i] + boost, matched };
  });

  scored.sort((a, b) => b.adjustedScore - a.adjustedScore);

  const passing = scored
    // 關鍵字完全命中的片段一定保留；沒命中關鍵字的片段才用語意分數把關，
    // 避免完全不相關的內容單靠語意分數的雜訊擠進結果
    .filter((s) => s.matched || s.baseScore >= SIM_THRESHOLD);

  const topScore = passing.length ? passing[0].adjustedScore : 0;
  const scoreFloor = topScore * RESULT_RELATIVE_CUTOFF;

  const results = passing
    .filter((s) => s.adjustedScore >= scoreFloor)
    .slice(0, RESULT_MAX_COUNT)
    .map((s) => ({
      title: s.doc.title,
      url: s.doc.url,
      text: s.doc.text,
      score: s.adjustedScore,
    }));

  renderResults(results, queryWords);
}

function onQueryChanged(event) {
  // 阻止 mkdocs 內建的本地搜尋 (worker) 接手同一個輸入事件
  event.stopImmediatePropagation();

  const query = event.target.value.trim();
  clearTimeout(debounceTimer);

  if (!query) {
    setMeta(state.ready ? "輸入關鍵字開始搜尋" : "AI 搜尋引擎啟動中，請稍候…");
    setList("");
    return;
  }

  debounceTimer = setTimeout(() => runSearch(query), DEBOUNCE_MS);
}

function onFocus(event) {
  event.stopImmediatePropagation();
  const query = event.target.value.trim();
  if (query) {
    runSearch(query);
  } else if (!state.ready) {
    showLoadingStatus();
  }
}

async function init() {
  const input = getInput();
  if (!input) return;

  // 用 capture 階段攔截，並在處理常式內呼叫 stopImmediatePropagation()，
  // 讓 mkdocs 內建搜尋的事件處理常式不會再被觸發，藉此改造既有搜尋框
  input.addEventListener("input", onQueryChanged, true);
  input.addEventListener("focus", onFocus, true);

  showLoadingStatus();

  try {
    await Promise.all([loadModel(), loadDocsDatabase()]);
    state.ready = true;

    const query = input.value.trim();
    if (query) {
      runSearch(query);
    } else {
      setMeta("輸入關鍵字開始搜尋");
    }
  } catch (err) {
    console.error(err);
    setMeta("AI 搜尋引擎啟動失敗，請重新整理頁面再試一次");
  }
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}
