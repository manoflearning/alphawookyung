#include <bits/stdc++.h>
#define ll long long
#define pii pair<int, int>
#define pll pair<ll, ll>
using namespace std;

char arr[10][10]; 

inline char trans(char x);

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int T; cin >> T;

	while (T--) {
		for (int y = 1; y <= 9; y++) {
			for (int x = 1; x <= 9; x++) {
				cin >> arr[y][x];
			}
		}

		arr[1][1] = trans(arr[1][1]);
		arr[4][2] = trans(arr[4][2]);
		arr[7][3] = trans(arr[7][3]);

		arr[2][4] = trans(arr[2][4]);
		arr[5][5] = trans(arr[5][5]);
		arr[8][6] = trans(arr[8][6]);

		arr[3][7] = trans(arr[3][7]);
		arr[6][8] = trans(arr[6][8]);
		arr[9][9] = trans(arr[9][9]);

		for (int y = 1; y <= 9; y++) {
			for (int x = 1; x <= 9; x++) {
				cout << arr[y][x];
			}
			cout << endl;
		}
		cout << endl;
	}

	return 0;
}

inline char trans(char x) {
	return x == '9' ? '1' : x + 1;
}