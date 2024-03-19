#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

int N, par[55];
vector<int> child[55];

int f(int now);

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	cin >> N;
	for (int i = 0; i < N; i++) {
		int p; cin >> p;
		par[i] = p;
		if (p != -1) child[p].push_back(i);
	}

	cout << f(0);

	return 0;
}

int f(int now) {
	vector<int> time;

	for (int next : child[now]) {
		time.push_back(f(next));
	}

	sort(time.rbegin(), time.rend());

	int ret = 0;
	for (int i = 0; i < time.size(); i++) {
		ret = max(ret, time[i] + i + 1);
	}
	
	return ret;
}