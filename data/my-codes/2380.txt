#include <bits/stdc++.h>
#define ll long long
#define pii pair<int, int>
#define pli pair<ll, int>
#define pil pair<int, ll>
#define pll pair<ll, ll>
using namespace std;

pii arr[105];

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int T; cin >> T;

	while (T--) {
		int N; cin >> N;

		bool YES = true;
		cin >> arr[0].first >> arr[0].second;

		if (arr[0].first < arr[0].second) YES = false;

		for (int i = 1; i < N; i++) {
			cin >> arr[i].first >> arr[i].second;

			if (arr[i - 1].first > arr[i].first) YES = false;
			if (arr[i - 1].second > arr[i].second) YES = false;
			if (arr[i].first - arr[i - 1].first < arr[i].second - arr[i - 1].second) YES = false;
			if (arr[i].first < arr[i].second) YES = false;
		}

		if (YES) cout << "YES\n";
		else cout << "NO\n";
	}

	return 0;
}