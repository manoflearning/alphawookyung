#include <iostream>
using namespace std;

int main() {
	int t;
	scanf("%d", &t);

	for (int test = 0; test < t; test++) {
		int hp, mp, atk, def, _hp, _mp, _atk, _def;

		scanf("%d %d %d %d %d %d %d %d", &hp, &mp, &atk, &def, &_hp, &_mp, &_atk, &_def);

		hp += _hp;
		mp += _mp;
		atk += _atk;
		def += _def;

		if (hp < 1) hp = 1;
		if (mp < 1) mp = 1;
		if (atk < 0) atk = 0;

		printf("%d\n", hp + 5 * mp + 2 * atk + 2 * def);
	}

	return 0;
}