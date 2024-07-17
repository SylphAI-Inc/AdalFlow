// _static/js/custom.js
document.addEventListener("DOMContentLoaded", function() {
    var closeButton = document.getElementById("close-banner");
    var banner = document.getElementById("announcement-banner");

    var bannerClosed = localStorage.getItem('bannerClosed');

    if (bannerClosed === 'true') {
        banner.style.display = 'none';
    } else {
        banner.style.display = 'flex';
    }

    if (closeButton && banner) {
        closeButton.addEventListener("click", function() {
            banner.style.display = "none";
            localStorage.setItem('bannerClosed', 'true');
        });
    }
});
