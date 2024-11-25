document.addEventListener('DOMContentLoaded', () => {
    const toggleSwitch = document.getElementById('mode-toggle');
    const body = document.body;

    // Load saved dark mode preference
    const isDarkMode = localStorage.getItem('darkMode') === 'true';
    toggleSwitch.checked = isDarkMode;
    if (isDarkMode) {
        body.classList.add('dark-mode');
    }

    // Toggle dark/light mode
    toggleSwitch.addEventListener('change', () => {
        if (toggleSwitch.checked) {
            body.classList.add('dark-mode');
            localStorage.setItem('darkMode', 'true');
        } else {
            body.classList.remove('dark-mode');
            localStorage.setItem('darkMode', 'false');
        }
    });
});
