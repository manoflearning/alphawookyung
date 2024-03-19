#include <iostream>
#include <vector>
#include <algorithm>
#include <queue>
#define pii pair<int, int>
using namespace std;

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int N; cin >> N;

	vector<pii> arr(N);
	for (auto& i : arr) 
		cin >> i.first >> i.second;

	sort(arr.begin(), arr.end());

	priority_queue<int, vector<int>, greater<int>> pq;
	pq.push(arr[0].second);

	int ans = 0;
	for (int i = 1; i < N; i++) {
		while (!pq.empty() && pq.top() <= arr[i].first) pq.pop();
		
		if (!pq.empty())
			pq.push(arr[i].second);

		ans = max(ans, (int)pq.size());
	}

	cout << ans << '\n';

	return 0;
}