// Accordion behavior for model menus - only one menu open at a time
document.addEventListener('DOMContentLoaded', function() {
    const triggers = document.querySelectorAll('.model-trigger');
    
    triggers.forEach(trigger => {
        trigger.addEventListener('change', function() {
            // If this checkbox is being checked
            if (this.checked) {
                // Uncheck all other checkboxes in the same container
                const container = this.closest('.interactive-map-container');
                if (container) {
                    const otherTriggers = container.querySelectorAll('.model-trigger');
                    otherTriggers.forEach(otherTrigger => {
                        if (otherTrigger !== this) {
                            otherTrigger.checked = false;
                        }
                    });
                }
            }
        });
    });
});
