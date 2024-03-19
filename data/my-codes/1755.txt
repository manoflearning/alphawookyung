#include <bits/stdc++.h>
using namespace std;

const char ACGT[] = { 'A', 'C', 'G', 'T' };

int n;
string s;
int cnt[4];

void print(int res1, char res2) {
    cout << res1 << '\n';
    for (int i = 0; i < n; i++) cout << res2;
}

int main() {
    cin.tie(NULL); cout.tie(NULL);
    ios_base::sync_with_stdio(false);

    cin >> n >> s;

    for (auto& i : s) {
        if (i == 'A') cnt[0]++;
        else if (i == 'C') cnt[1]++;
        else if (i == 'G') cnt[2]++;
        else if (i == 'T') cnt[3]++;
    }

    for (int i = 0; i < 4; i++) {
        int isAns = 1;
        for (int j = 0; j < 4; j++)
            isAns &= (cnt[i] <= cnt[j]);
        if (isAns) {
            print(cnt[i], ACGT[i]); break;
        }
    }
}