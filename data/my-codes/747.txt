#include <iostream>
#include <vector>
#include <algorithm>
#include <map>
using namespace std;
#define pii pair<int, int>

const int MAX = 6 * 1e5;

int N, psum[MAX + 5];
map<int, int> mp, revMp;
vector<pii> arr;

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	cin >> N;

	arr.resize(N);
	for (auto& i : arr) {
		cin >> i.first >> i.second;
		mp.insert({ i.first, 0 });
		mp.insert({ i.second, 0 });
	}

	int cnt = 0;
	for (auto& i : mp) {
		i.second = ++cnt;
		revMp.insert({ i.second, i.first });
	}

	for (auto& i : arr) {
		psum[mp.find(i.first)->second]++;
		psum[mp.find(i.second)->second + 1]--;
	}

	for (int i = 2; i <= cnt; i++)
		psum[i] += psum[i - 1];

	int ans = 0;
	for (int i = 1; i <= cnt; i++) { 
		if (psum[ans] < psum[i]) ans = i;
	}
	
	cout << psum[ans] << '\n';

	int flag = revMp.find(ans)->second;
	for (int i = 0; i < arr.size(); i++) {
		if (arr[i].first <= flag && flag <= arr[i].second) cout << i + 1 << ' ';
	}

	return 0;
}