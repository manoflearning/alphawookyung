#include <iostream>
#include <vector>
#define vvi vector<vector<int>>
using namespace std;

int N, M, K;
vvi A, B, C;

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	cin >> N >> M;

	A.resize(N, vector<int>(M));
	for (int n = 0; n < N; n++) {
		for (int m = 0; m < M; m++) {
			cin >> A[n][m];
		}
	}

	cin >> M >> K;

	B.resize(M, vector<int>(K));
	for (int m = 0; m < M; m++) {
		for (int k = 0; k < K; k++) {
			cin >> B[m][k];
		}
	}

	C.resize(N, vector<int>(K));
	for (int n = 0; n < N; n++) {
		for (int k = 0; k < K; k++) {
			for (int m = 0; m < M; m++) {
				C[n][k] += A[n][m] * B[m][k];
			}
		}
	}

	for (int n = 0; n < N; n++) {
		for (int k = 0; k < K; k++) {
			cout << C[n][k] << ' ';
		}
		cout << '\n';
	}

	return 0;
}