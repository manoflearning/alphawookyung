#include <iostream>
#include <vector>
using namespace std;

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int N; cin >> N;

	vector<int> ans(N);

	for (int h = 1; h <= N; h++) {
		int x; cin >> x;

		int cnt = 0;
		for (int& i : ans) {
			if (i == 0) cnt++;
			if (cnt == x + 1) {
				i = h;
				break;
			}
		}
	}
	
	for (int& i : ans) cout << i << ' ';

	return 0;
}