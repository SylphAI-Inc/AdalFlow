document.addEventListener('DOMContentLoaded', function () {
    var dismissed = localStorage.getItem('announcement-dismissed');
    var banner = document.getElementById('announcement-banner');
    
    if (!dismissed) {
        banner.style.display = 'block';
    } else {
        banner.style.display = 'none';
    }
});

function dismissBanner() {
    var banner = document.getElementById('announcement-banner');
    banner.style.display = 'none';
    localStorage.setItem('announcement-dismissed', 'true');
}
