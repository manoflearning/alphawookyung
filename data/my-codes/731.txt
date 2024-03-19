#include <iostream>
#include <algorithm>
using namespace std;

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int N, K, P, W;
	cin >> N >> K >> P >> W;

	cout << P / W + (P % W ? 1 : 0);

	return 0;
}