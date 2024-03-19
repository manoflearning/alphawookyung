#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int N; cin >> N;

	vector<int> q(N);
	for (int& i : q) cin >> i;

	vector<int> sq = q, label(2001);
	sort(sq.begin(), sq.end());
	for (int i = 0; i < N; i++)
		label[sq[i] + 1000] = i;
	
	vector<bool> visited(N);

	int ans = 0;
	for (int i : q) {
		int pos = label[i + 1000];
		if (0 < pos && visited[pos - 1]);
		else if (pos < N - 1 && visited[pos + 1]);
		else ans++;

		visited[pos] = true;
	}

	cout << ans << '\n';

	return 0;
}