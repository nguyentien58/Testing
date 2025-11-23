#include <stdio.h>
#include <stdbool.h>
#include <math.h>
//Kiem tra so chan va le
void kiem_tra_so_chan_le(int n) {
    if(n % 2 == 0) {
        printf("Day la so chan\n");
    }
    else {
        printf("Day la so le\n");
    }
}
//Kiem tra so duong va am
void kiem_tra_so_am_duong(int n) {
    if(n<0) {
        printf("Day la so am\n");
        //ket thuc khi dk dung
    }
    else {
        printf("Day la so duong\n");
        //ket thuc khi dk dung
    }
}
bool kiem_tra_so_nguyen_to(int n) {
    if(n<0) {
        printf("Day khong phai la so nguyen to\n");
        return false;
        //Tra ve flase khi dieu kien sai
    }
    for(int i=2;i<=sqrt(n);i++) {
        if (i%2==0) {
            printf("Day khong phai so nguyen to\nVi %d chia het cho 2",n);
            return false;
        //Tra ve flase khi dieu kien sai
        }  
    }
    printf("Day la so nguyen to");
    return true;
        //Ket qua no se tra ve true
}
int count;
bool kiem_tra_co_may_chu_so(int n) {
    if(n==0) {
        printf("N co 1 chu so");
        return false;
    }
    else if(n <10 || n>99) {
        printf("\nYeu cau nhap phai la so co hai chu so\n");
        //dieu kien cua bai la n co 2 chu so
        return false;
    }
    while(n>0) {
        n = n/10;
        count++;
        //tang so dem moi khi chia duoc cho 10
    }
       printf("\nN co %d chu so",count); //xuat ra man hinh tong so lan chia duoc
    }
    void tim_so_dao_nguoc(int n) {
        int so_dao_nguoc = 0;
    while(n>0) {
        int chu_so_cuoi = n %10;
        so_dao_nguoc = so_dao_nguoc * 10 + chu_so_cuoi;
        n = n /10;
        //dong lenh chay cho den khi dung
    }
            printf("So dao nguoc cua ban la %d",so_dao_nguoc);
    }
int main() {
    int N;
    do {
    printf("Nhap gia tri N ma ban muon kiem tra: ");
    if((scanf("%d",&N)!=1)){
        printf("YEU cau nhap so\nYeu cau ban nhap lai.\n");
        while(getchar() != '\n');
    }else {
        break;
    }
    }while(1);//Vong lap selap vo tan, break khi nhap dung ket qua
    kiem_tra_so_chan_le(N);
    kiem_tra_so_am_duong(N);
    kiem_tra_so_nguyen_to(N);
    kiem_tra_co_may_chu_so(N);
    tim_so_dao_nguoc(N);
    //kim tra phep tinh

}
