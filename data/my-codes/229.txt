#include <iostream>
using namespace std;

class quiz {
protected:
	string ox;
	int length{ 0 }, score{ 0 }, num{ 1 };
public:
	void cal() {
		cin >> ox;
		length = ox.length();

		for (int i{ 0 }; i < length; i++) {
			if (ox[i] == 'X')
				num = 1;
			else {
				score += num;
				num++;
			}
		}

		cout << score;
	}
};

int main() {
	int n;
	cin >> n;
	quiz* Q = new quiz[n];
	for (int i{ 0 }; i < n; i++) {
		Q[i].cal();
		cout << '\n';
	}

	return 0;
}