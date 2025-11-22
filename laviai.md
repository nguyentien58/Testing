
#include <stdio.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
static void msleep(int ms) { Sleep(ms); }
#else
#include <unistd.h>
static void msleep(int ms) { usleep(ms * 1000); }
#endif

// In tung ky tu cham cham
void print_slow(const char *s, int char_delay_ms)
{
    for (size_t i = 0; i < strlen(s); i++)
    {
        putchar(s[i]);
        fflush(stdout);
        msleep(char_delay_ms);
    }
}

int main(void)
{

    const char *phrases[] = {
        "Cu buoc tiep di, anh khong niu dau",
        "Anh biet dau ma, you know",
        "Song nhu khi anh chua gap em, nhu hai ta chua tung thieu nhau",
        "Chua bao gio anh chan em, day dau phai la van game",
        "Cau yeu thuong nay qua quen, em doan xem",
        "Gio lai ngoi o trong phong thu, va anh lai cuon (Dua anh to draw)",
        "Dan long rang phai giu tinh tao, nen anh phai uong (Chut anh phai lo)",
        "Bat bai nhac em thich ngay xua, cam xuc lai tuon (Dau de danh cho)",
        "Tu hoi ban than sao ngay do tay em lai buong"};

    // delay giữa từng câu — chỉnh tay theo flow LAVIAI (75–80 BPM)
    int delays_ms[] = {
        1000, // Cu buoc tiep di, anh khong niu dau
        1100, // Anh biet dau ma, you know
        1300, // Song nhu khi anh chua gap em...
        1130, // Chua bao gio anh chan em...
        1100, // Cau yeu thuong nay qua quen...
        1200, // Gio lai ngoi o trong phong thu...
        1300, // Dan long rang phai giu tinh tao...
        1300, // Bat bai nhac em thich ngay xua...
        1500  // Tu hoi ban than sao ngay do...
    };

    int n = sizeof(phrases) / sizeof(phrases[0]);

    int char_delay_ms = 22;        // Delay giữa từng ký tự (giống tốc độ rap nhẹ)
    double tempo_multiplier = 1.0; // chỉnh 0.8 nhanh hơn, 1.2 chậm hơn

    puts("---- ACTION ----\n");

    for (int i = 0; i < n; i++)
    {
        print_slow(phrases[i], char_delay_ms);
        putchar('\n');
        fflush(stdout);

        int wait = (int)(delays_ms[i] * tempo_multiplier);
        msleep(wait);
    }

    puts("\n---- LA VI AI?????????? ----");
    return 0;
}
