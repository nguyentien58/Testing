#include <stdio.h>
int main()
{
	int d,m,y;
	printf("Nhap ngay cua ban: ");
		if(scanf("%d",&d)!=1){
	
			printf("Loi nhap lieu");
			return 1;
	}
	printf("Nhap thang cua ban: ");
		if(scanf("%d",&m)!=1){
		
			printf("Loi nhap lieu");
			return 1;
	}
	printf("Nhap nam cua ban: ");
		if(scanf("%d",&y)!=1){
			printf("Loi nhap lieu");
			return 1;
	}
	int leapyear = ((y%4 == 0 && y % 100 !=0)|| (y % 400 == 0));
	switch(m-1){
	case 11:
		d += 30;
	case 10:
		d += 31;
	case 9:
		d += 30;
	case 8:
		d += 31;
	case 7:
		d += 31;
	case 6:
		d += 30;
	case 5:
		d += 31;
	case 4:
		d += 30;
	case 3:
		d += 31;
	case 2:
	
	d += leapyear ? 29: 28;
			
	case 1:
		d += 31;		
	case 0:
		break; 
	default:
		printf("Thang khong hop le!");
		return 1;
	
}
if(leapyear){
	printf("Day la nam nhuan %d / 366",d);
}
else {
	printf("Day la nam khong nhuan %d / 365",d);	
	
}
}
