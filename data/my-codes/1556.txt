#include <iostream>
#include <algorithm>
#include <vector>
using namespace std;

int N, M;
vector<int> arr, ans;

void make(int flag, int digit);

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	cin >> N >> M;

	for (int i = 0; i < N; i++) {
		int x; cin >> x;
		arr.push_back(x);
	}

	sort(arr.begin(), arr.end());

	make(0, 0);

	return 0;
}

void make(int flag, int digit) {
	if (digit == M) {
		for (int x : ans) cout << x << ' ';
		cout << '\n';
		return;
	}

	for (int i = flag; i < N; i++) {
		ans.push_back(arr[i]);
		make(i, digit + 1);
		ans.pop_back();
	}
}