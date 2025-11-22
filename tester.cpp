#include <stdio.h>

// Hàm tính tổng các chữ số của một số
int tinh_tong_chu_so(int n) {
    int tong = 0;
    while (n > 0) {
        tong += n % 10; // Lấy chữ số cuối cùng
        n /= 10;        // Loại bỏ chữ số cuối cùng
    }
    return tong;
}

// Hàm tính Con số Chủ đạo
int tinh_con_so_chu_dao(int ngay, int thang, int nam) {
    // Bước 1: Rút gọn riêng biệt từng phần (Ngày, Tháng, Năm)

    // Rút gọn Ngày
    int tong_ngay = ngay;
    while (tong_ngay > 9 && tong_ngay != 11 && tong_ngay != 22) {
        tong_ngay = tinh_tong_chu_so(tong_ngay);
    }

    // Rút gọn Tháng
    int tong_thang = thang;
    while (tong_thang > 9 && tong_thang != 11 && tong_thang != 22) {
        tong_thang = tinh_tong_chu_so(tong_thang);
    }

    // Rút gọn Năm
    int tong_nam = nam;
    // Rút gọn đến 1 chữ số hoặc số Master (11, 22, 33)
    while (tong_nam > 9 && tong_nam != 11 && tong_nam != 22 && tong_nam != 33) {
        tong_nam = tinh_tong_chu_so(tong_nam);
    }

    // Bước 2: Cộng tổng các phần đã rút gọn
    int tong_cuoi_cung = tong_ngay + tong_thang + tong_nam;

    // Bước 3: Rút gọn tổng cuối cùng đến Con số Chủ đạo (1-9, 11, 22, 33)
    while (tong_cuoi_cung > 9 && tong_cuoi_cung != 11 && tong_cuoi_cung != 22 && tong_cuoi_cung != 33) {
        tong_cuoi_cung = tinh_tong_chu_so(tong_cuoi_cung);
    }

    return tong_cuoi_cung;
}

int main() {
    int ngay, thang, nam;

    printf("--- Tinh Con so Chu dao (Life Path Number) ---\n");
    printf("Nhap ngay sinh (vi du: 15): ");
    if (scanf("%d", &ngay) != 1) return 1;
    printf("Nhap thang sinh (vi du: 8): ");
    if (scanf("%d", &thang) != 1) return 1;
    printf("Nhap nam sinh (vi du: 2005): ");
    if (scanf("%d", &nam) != 1) return 1;

    // Kiem tra du lieu dau vao co hop le khong (toi thieu)
    if (ngay < 1 || ngay > 31 || thang < 1 || thang > 12 || nam < 1900) {
        printf("Ngay thang nam khong hop le.\n");
        return 1;
    }

    int con_so_chu_dao = tinh_con_so_chu_dao(ngay, thang, nam);

    printf("\nNgay sinh: %d/%d/%d\n", ngay, thang, nam);
    printf("Con so Chu dao cua ban la: **%d**\n", con_so_chu_dao);
//Y nghia con so chu dao cua ban
    switch(con_so_chu_dao){
        case 1:
        printf("Lanh dao, Doc lap, Tien phong");
        break;
        case 2:
        printf("Hoa giai, Hop tac, Nhay cam");
        break;
        case 3:
        printf("Sang tao, Giao tiep, Lac quan");
        break;
        case 4:
        printf("Ky luat, Logic, Thuc te");
        break;
        case 5:
        printf("Tu do, Linh hoat, Phieu luu");
        break;
        case 6:
        printf("Trach nhiem, Yeu thuong, Quan tam");
        break;
        case 7:
        printf("Noi tam, Tri ly, Phan tich");
        break;
        case 8:
        printf("Tham vong, Quyen luc, Tai chinh");
        break;
        case 9:
        printf("Nhan dao, Ly tuong, Vi tha");
        break;
        case 11:
        printf("Truc giac, Cam hung tinh than, Nguoi nhin xa trong rong");
        break;
        case 22:
        printf("Nha Kien tao, Bien y tuong thanh hien thuc");
        break;
        case 33 :
        printf("Tinh yeu vo dieu kien, Dan dat tam linh");
        break;
    }


    return 0;
    }
