#include <iostream>
#include <random>
#include <vector>
#include <Eigen/Dense>
#include <sstream>
#include <fstream>

class Perceptron {
public:
	double eta; // wwspółczynnik uczenia
	int n_iter; //Liczba przebiegów
	int random_state;  //ziarno generatora
	Eigen::VectorXd w_; // Wagi
	std::vector<int> errors_; // Lista błędów w każdej epoce/iteracji

	Perceptron(double eta = 0.01, int n_iter = 50, int random_state = 1) : eta(eta), n_iter(n_iter), random_state(random_state)
	{
	}

	// metoda dopasowania
	void fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) {
		// X - wektory uczące 
		// Y - wartości docelowe
		std::mt19937 gen(random_state);// inicjalizacja wag
		std::normal_distribution<double> dist(0.0, 0.01);

		w_ = Eigen::VectorXd::Zero(X.cols() + 1); // wagi po dopasowaniu
		for (int i = 0; i < w_.size(); ++i) {
			w_(i) = dist(gen);
		}

		errors_.clear();

		for (int i = 0; i < n_iter; ++i)
		{
			int errors = 0;
			for (int j = 0; j < X.rows(); ++j) {
				Eigen::VectorXd xi = X.row(j); // pomocniczy wektro cech
				double target = y(j);

				// wartosc aktualizacji wagi
				double update = eta * (target - predict(xi));

				// aktualizacja wagi 
				w_.segment(1, xi.size()) += update * xi;
				w_(0) += update;

				errors += (update != 0.0) ? 1 : 0;
			}
			errors_.push_back(errors);
		}
	}

	//Obliczanie pubudzenia
	// Funkcja która policzy petencjał poprzez iloczyn skalarny wag oraz cech
	double net_input(const Eigen::VectorXd& X) {
		// funbkcja segment w bibliotece eigen powoduje skrocenei wektora wag 
		// o jeden poniewaz poprzednio posiadał on o jeden element wiecej czyli wage początkową
		return X.dot(w_.segment(1, X.size())) + w_(0);
	}

	//Przewidywanie etykiety
	//Decyzja czy percepton zostanie pobudzony czy tez nie jezeli pubudzenie 
	//bedzie rowne lub wieksze od 0 to przektorzy ten prog pobudzenia i p[rzypisze etykiete 1 
	int predict(const Eigen::MatrixXd& X) {
		return (net_input(X) >= 0.0) ? 1 : -1;
	} 
};

class FileReader {
public:
	bool load(std::string filename, Eigen::MatrixXd& X, Eigen::VectorXd& y) {
		std::ifstream file(filename);
		if (!file.is_open()) return false;


		std::vector<std::vector<double>> features;
		std::vector<double> labels;
		std::string line;

		std::string header;
		std::getline(file, header);

		while (std::getline(file, line)) {
			if (line.empty()) continue;

			std::stringstream ss(line);
			std::string val;
			std::vector<double> row;

			std::getline(ss, val, ',');

			for (int i = 0; i < 4; ++i) {
				std::getline(ss, val, ',');
				row.push_back(std::stod(val));
			}

			std::getline(ss, val, ',');
			if (val == "Iris-setosa") {
				labels.push_back(-1);
				features.push_back(row);
			}
			else if (val == "Iris-versicolor") {
				labels.push_back(1);
				features.push_back(row);
			}
		}

		X.resize(features.size(), 4);
		y.resize(labels.size());
		for (int i = 0; i < features.size(); ++i) {
			for (int j = 0; j < 4; ++j) X(i, j) = features[i][j];
			y(i) = labels[i];
		}
		return true;
	}
};

int main() {
	Eigen::MatrixXd X;
	Eigen::VectorXd y;
	std::ifstream test("Iris.csv");
	if (test.is_open()) {
		std::cout << "Hurra! Plik znaleziony!" << std::endl;
	}
	else {
		std::cout << "Nie widze pliku Iris.csv w folderze roboczym." << std::endl;
	}
	FileReader reader;
	bool loaded = reader.load("Iris.csv", X, y);

	if (!loaded) {
		std::cerr << "Nie udalo sie otworzyc pliku iris.data!" << std::endl;
		return 1;
	}
	std::cout << "Wczytano " << X.rows() << " przykladow." << std::endl;

	Perceptron ppn(0.1, 10);
	ppn.fit(X, y);

	std::cout << "Trening zakonczony. Bledy w kolejnych epokach:" << std::endl;
	for (int i = 0; i < ppn.errors_.size(); ++i) {
		std::cout << "Epoka " << i + 1 << ": " << ppn.errors_[i] << " bledow" << std::endl;
	}


	return 0;
}