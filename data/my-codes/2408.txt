#include <iostream>
#include <vector>
using namespace std;

const int MAX = 1e5;

struct nn {
	int n1, n2;
};

vector<int> adj[MAX + 5];
vector<nn> Q;

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int n; cin >> n;

	for (int i = 0; i < n - 1; i++) {
		int a, b; cin >> a >> b;
		adj[a].push_back(b);
		adj[b].push_back(a);
		Q.push_back({ a, b });
	}

	bool isBamboo = true;
	int pNode;
	for (int i = 1; i <= n; i++) {
		if (adj[i].size() >= 3) {
			isBamboo = false;
			pNode = i;
			break;
		}
	}

	if (isBamboo) {
		for (int i = 0; i < n - 1; i++)
			cout << i << '\n';
	}
	else {
		int flag = 0, nflag = 3;
		for (int i = 0; i < n - 1; i++) {
			if (flag <= 2 && (Q[i].n1 == pNode || Q[i].n2 == pNode)) {
				cout << flag << '\n';
				flag++;
			}
			else {
				cout << nflag << '\n';
				nflag++;
			}
		}
	}

	return 0;
}