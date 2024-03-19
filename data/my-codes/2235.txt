#include <iostream>
#include <vector>
#include <cstring>
#include <algorithm>
#define ll long long
using namespace std;

string s;
int b[101], ans[201];
bool exist[201];

bool cmp(int a, int b) {
	return a > b;
}

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int T;
	cin >> T;

	while (T--) {
		memset(b, 0, sizeof(b));
		memset(ans, 0, sizeof(ans));
		memset(exist, 0, sizeof(exist));

		int n;
		cin >> n;

		for (int i = 1; i <= n; i++) {
			cin >> b[i];
			exist[b[i]] = true;
		}

		bool YES = true;
		for (int i = 1; i <= n; i++) {
			ans[2 * i - 1] = b[i];

			if (b[i] == 2 * n) {
				YES = false;
				break;
			}
			for (int l = b[i] + 1; l <= 2 * n; l++) {
				if (!exist[l]) {
					ans[2 * i] = l;
					exist[l] = true;
					break;
				}
				if (l == 2 * n) {
					YES = false;
					i = n + 1;
					break;
				}
			}
		}

		if (!YES) cout << -1 << '\n';
		else {
			for (int i = 1; i <= 2 * n; i++)
				cout << ans[i] << ' ';
			cout << '\n';
		}
	}

	return 0;
}