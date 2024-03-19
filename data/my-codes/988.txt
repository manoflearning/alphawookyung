#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
using namespace std;
#define ll long long

int N;
vector<ll> arr;

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	cin >> N;

	arr.resize(N);
	for (auto& i : arr) cin >> i;

	sort(arr.begin(), arr.end());

	ll mn = 1e10;
	vector<ll> ans;

	for (int i = 0; i < arr.size(); i++) {
		for (int l = i + 1; l < arr.size(); l++) {
			ll sum = arr[i] + arr[l];
			
			int val[2];

			val[0] = lower_bound(arr.begin(), arr.end(), -sum) - arr.begin();
			while (val[0] < arr.size() && (val[0] == i || val[0] == l)) val[0]++;
			val[1] = val[0] - 1;
			while (0 <= val[1] && (val[1] == i || val[1] == l)) val[1]--;

			for (int k = 0; k < 2; k++) {
				if (val[k] < arr.size() && !(val[k] == i || val[k] == l)) {
					if (mn > fabs(sum + arr[val[k]])) {
						mn = fabs(sum + arr[val[k]]);

						ans.clear();
						ans.push_back(arr[i]);
						ans.push_back(arr[l]);
						ans.push_back(arr[val[k]]);
					}
				}
			}
		}
	}

	sort(ans.begin(), ans.end());
	for (int i = 0; i < ans.size(); i++) 
		cout << ans[i] << ' ';

	return 0;
}