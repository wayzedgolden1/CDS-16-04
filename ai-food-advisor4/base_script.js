 // base_script.js

// Hàm cập nhật Navbar dựa trên trạng thái đăng nhập
async function updateNavbar() {
    const res = await fetch('/api/status');
    const status = await res.json();
    const authLink = document.getElementById('auth-link');
    const authNavItem = document.getElementById('auth-nav-item');
    const profileNavItem = document.getElementById('profile-nav-item');

    if (status.logged_in) {
        authLink.textContent = 'Đăng Xuất (' + status.username + ')';
        authLink.href = '/api/logout';
        profileNavItem.style.display = 'block';

        // Logic chuyển hướng/hiển thị thông báo
        if (window.location.pathname === '/auth') {
            window.location.href = status.has_profile ? '/' : '/profile';
            return status;
        }

        if (window.location.pathname !== '/profile' && !status.has_profile) {
            // Thông báo trên trang chủ nếu chưa có hồ sơ
            const container = document.querySelector('.container');
            const currentAlert = document.getElementById('profile-alert');
            if (!currentAlert) {
                const alertDiv = document.createElement('div');
                alertDiv.id = 'profile-alert';
                alertDiv.className = 'alert alert-warning text-center mt-3';
                alertDiv.innerHTML = '<strong>Xin chào, ' + status.username + '!</strong> Bạn chưa có Hồ sơ. Vui lòng <a href="/profile" class="alert-link">Nhập Hồ sơ cá nhân</a> để sử dụng đầy đủ tính năng.';
                container.prepend(alertDiv);
            }
        }
    } else {
        authLink.textContent = 'Đăng Nhập/Đăng Ký';
        authLink.href = '/auth';
        profileNavItem.style.display = 'none';

        // Nếu chưa đăng nhập mà ở trang bảo mật, chuyển về auth
        if (window.location.pathname === '/profile' || window.location.pathname === '/') {
            // Chỉ cho phép ở trang chủ khi chưa đăng nhập, nhưng sẽ hiện nút "Vui lòng đăng nhập"
            if (window.location.pathname === '/profile') {
                window.location.href = '/auth';
            }
        }
    }
    return status;
}

document.addEventListener('DOMContentLoaded', updateNavbar);   