#include <iostream>
using namespace std;

int h, m, s, togo;

void splus();
void mplus();
void hplus();

int main() {  
    scanf("%d %d %d", &h, &m, &s);
    
    scanf("%d", &togo);
    
    while(togo > 0) splus();
    
    printf("%d %d %d", h, m, s);
    
    return 0;
}

void splus() {
    togo--;
    
    if (s < 59) s++;
    else {
        s = 0;
        mplus();
    }
}
void mplus() {
    if (m < 59) m++;
    else {
        m = 0,
        hplus();
    }
}
void hplus() {
    if (h < 23) h++;
    else h = 0;
}