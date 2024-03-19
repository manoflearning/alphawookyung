#include <bits/stdc++.h>
using namespace std;
#define ll long long
#define pii pair<int, int>
#define pll pair<ll, ll>

const int INF = 1e9 + 7;
const int MOD = 1e9 + 7;
const int dy[] = { 0, 0, 1, -1 };
const int dx[] = { 1, -1, 0, 0 };
const long double PI = 3.141592653589793238462643383279502884197;

int A[305][305];
int R[305], C[305];

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int tc; cin >> tc;

	while (tc--) {
		memset(A, 0, sizeof(A));
		memset(R, 0, sizeof(R));
		memset(C, 0, sizeof(C));

		int n, k;
		cin >> n >> k;

		int div = k / n, left = k % n;
		for (int y = 1; y <= n; y++) {
			int cnt = div, x = y;
			while (cnt--) {
				A[y][x] = 1;

				x = ((x - 1) + 1) % n + 1;
			}
		}

		for (int y = 1; y <= left; y++) {
			int x = ((y - 1) + div) % n + 1;
			
			A[y][x] = 1;
		}

		for (int y = 1; y <= n; y++) {
			for (int x = 1; x <= n; x++) {
				R[y] += A[y][x];
				C[x] += A[y][x];
			}
		}

		int mnR = INF, mxR = 0, mnC = INF, mxC = 0;
		for (int i = 1; i <= n; i++) {
			mnR = min(mnR, R[i]);
			mxR = max(mxR, R[i]);
			mnC = min(mnC, C[i]);
			mxC = max(mxC, C[i]);
		}

		cout << (mxR - mnR) * (mxR - mnR) + (mxC - mnC) * (mxC - mnC) << '\n';
		for (int y = 1; y <= n; y++) {
			for (int x = 1; x <= n; x++) {
				cout << A[y][x];
			}
			cout << '\n';
		}
	}

	return 0;
}