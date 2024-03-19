#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
using namespace std;

int N, ans;
vector<int> arr, arr2;
bool visited[10];

void f(int now);

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	cin >> N;

	arr.resize(N);
	for (int& i : arr) cin >> i;

	f(0);

	cout << ans << '\n';

	return 0;
}

void f(int now) {
	if (now == N) {
		int sum = 0;
		for (int i = 0; i < arr2.size() - 1; i++)
			sum += fabs(arr2[i] - arr2[i + 1]);
		ans = max(ans, sum);
		return;
	}

	for (int i = 0; i < arr.size(); i++) {
		if (visited[i]) continue;
		arr2.push_back(arr[i]);
		visited[i] = true;
		f(now + 1);
		arr2.pop_back();
		visited[i] = false;
	}
}