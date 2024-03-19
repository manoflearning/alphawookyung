#include <bits/stdc++.h>
#define ll long long
#define pii pair<int, int>
#define pll pair<ll, ll>
using namespace std;

const int MAX = 1e5;

ll arr[MAX + 5];

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);
	
	int T; cin >> T;

	while (T--) {
		int N; cin >> N;

		for (int i = 1; i <= N; i++) {
			cin >> arr[i];
		}

		int ans = 0;
		for (int i = 2; i <= N; i++) {
			if (arr[i - 1] > arr[i]) {
				ll mn = arr[i - 1] - arr[i];
				
				for (int l = 0; l <= 30; l++) {
					if (mn < ((ll)1 << l)) break;
					
					if (mn & ((ll)1 << l)) ans = max(ans, l + 1); 
				}

				arr[i] = arr[i - 1];
			}
		}

		cout << ans << endl;
	}

	return 0;
}