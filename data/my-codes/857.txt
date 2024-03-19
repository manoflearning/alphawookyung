#include <iostream>
#include <vector>
#include <cmath>
using namespace std;

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int n; cin >> n;

	vector<string> arr(n);
	for (auto& i : arr) cin >> i;

	int ans = 0;

	vector<int> cnt1(26);
	for (auto& c : arr[0]) cnt1[c - 'A']++;
	
	for (int i = 1; i < n; i++) {
		vector<int> cnt2(26);
		for (auto& c : arr[i]) cnt2[c - 'A']++;

		int one = 0, two = 0;
		for (int j = 0; j < 26; j++) {
			if (cnt1[j] > cnt2[j]) one += cnt1[j] - cnt2[j];
			if (cnt1[j] < cnt2[j]) two += cnt2[j] - cnt1[j];
		}

		if (one <= 1 && two <= 1) ans++;
	}

	cout << ans;

	return 0;
}