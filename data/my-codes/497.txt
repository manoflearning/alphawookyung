#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

int N, C, mid;
vector<int> arr, L, R;

void f(int idx, int sum);

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	cin >> N >> C;

	arr.resize(N);
	for (int& i : arr) cin >> i;

	if (N == 1) {
		if (arr[0] <= C) cout << 2;
		else cout << 1;
		return 0;
	}

	mid = (0 + N - 1) / 2;
	
	f(0, 0);
	f(mid + 1, 0);
	
	sort(L.begin(), L.end());
	sort(R.begin(), R.end());

	int ans = 0;
	for (int i = 0; i < L.size(); i++) {
		ans += upper_bound(R.begin(), R.end(), C - L[i]) - R.begin();
	}

	cout << ans;

	return 0;
}

void f(int idx, int sum) {
	if (C < sum) return;
	if (idx == mid || idx == N - 1) {
		if (idx == mid) {
			L.push_back(sum);
			L.push_back(sum + arr[idx]);
		}
		else {
			R.push_back(sum);
			R.push_back(sum + arr[idx]);
		}
		return;
	}
	
	f(idx + 1, sum);
	f(idx + 1, sum + arr[idx]);
}