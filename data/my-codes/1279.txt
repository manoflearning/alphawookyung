#include <iostream>
#include <vector>
using namespace std;

const int MAX = 2000;

bool isPalin[MAX + 5][MAX + 5];

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int N; cin >> N;

	vector<int> arr(N + 1);
	for (int i = 1; i < arr.size(); i++) cin >> arr[i];
	
	for (int i = 1; i <= N; i++) {
		isPalin[i][i] = true;
		if (i != N && arr[i] == arr[i + 1]) 
			isPalin[i][i + 1] = true;
	}

	for (int i = 3; i <= N; i++) {
		for (int L = 1; L + i - 1 <= N; L++) {
			int R = L + i - 1;

			if (isPalin[L + 1][R - 1] && arr[L] == arr[R]) isPalin[L][R] = true;
			else isPalin[L][R] = false;
		}
	}

	int M; cin >> M;
	while (M--) {
		int S, E;
		cin >> S >> E;

		cout << (isPalin[S][E] ? 1 : 0) << '\n';
	}

	return 0;
}