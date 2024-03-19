#include <bits/stdc++.h>
#define ll long long
#define pii pair<int, int>
#define pli pair<ll, int>
#define pil pair<int, ll>
#define pll pair<ll, ll>
using namespace std;

const int MAX = 1e5;

ll arr[MAX + 5];

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int T; cin >> T;

	while (T--) {
		int N, X;
		cin >> N >> X;

		for (int i = 0; i < N; i++)
			cin >> arr[i];

		sort(arr, arr + N);

		ll sum = 0;
		int ans = 0;
		for (int i = N - 1; i >= 0; i--) {
			sum += arr[i];

			if ((double)sum / (double)(N - i) >= X) ans = N - i;
		}

		cout << ans << '\n';
	}

	return 0;
}