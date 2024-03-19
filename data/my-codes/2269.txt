#include <iostream>
#include <cstring>
#include <cmath>
#define ll long long
using namespace std;

ll a[31], flag[100];
bool visited[100];

bool check(ll a, int f);

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int T; cin >> T;

	while (T--) {
		memset(visited, 0, sizeof(visited));

		int n, k, f;
		cin >> n >> k;

		flag[0] = 1, flag[1] = k;
		for (f = 2; flag[f - 1] * k <= (ll)1e16; f++)
			flag[f] = flag[f - 1] * k;
		f--;

		bool YES = true;
		for (int i = 1; i <= n; i++) {
			cin >> a[i];
			if (YES && !check(a[i], f)) {
				cout << "NO\n";
				YES = false;
			}
		}

		if (YES) cout << "YES\n";
	}

	return 0;
}

bool check(ll a, int f) {
	if (a == 0) return true;
	if (f == -1) return false;

	if (a >= flag[f]) {
		if (visited[f]) return false;
		else {
			visited[f] = true;
			check(a - flag[f], f - 1);
		}
	}
	else {
		check(a, f - 1);
	}
}