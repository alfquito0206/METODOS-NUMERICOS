"""
Programa de Metodos Numericos - Analisis Numerico y Programacion (ANP)
Estudiante: Alfredo S | Matricula: 2095320 | Semestre: 5 | Facultad: FCFM
"""

import numpy as np
import sympy as sp


class MetodosNumericos:
    def __init__(self):
        self.x = sp.Symbol('x')
        self.t = sp.Symbol('t')
        self.y = sp.Symbol('y')

    # ==================== BISECCION ====================
    def biseccion(self, func_str, a, b, tol=1e-6, max_iter=100):
        print("\n" + "=" * 60)
        print("METODO DE BISECCION")
        print("=" * 60)

        func_sym = sp.sympify(func_str)
        func = sp.lambdify(self.x, func_sym, 'numpy')

        print(f"Ecuacion: f(x) = {func_str}")
        print(f"Intervalo: [{a}, {b}] | Tolerancia: {tol}")

        fa, fb = func(a), func(b)

        if fa * fb > 0:
            print(f"ERROR: f({a})={fa:.6f} y f({b})={fb:.6f} tienen el mismo signo")
            return None

        print(f"\nf(a)*f(b) = {fa * fb:.6f} < 0 OK")
        print("\n{:<5} {:<15} {:<15} {:<15} {:<15}".format("Iter", "a", "b", "c", "f(c)"))
        print("-" * 70)

        for i in range(max_iter):
            c = (a + b) / 2
            fc = func(c)
            print(f"{i + 1:<5} {a:<15.8f} {b:<15.8f} {c:<15.8f} {fc:<15.8e}")

            if abs(fc) < tol or abs(b - a) < tol:
                print(f"\nRaiz: x = {c:.10f} | f(x) = {fc:.2e} | Iteraciones: {i + 1}")
                return c

            if fa * fc < 0:
                b, fb = c, fc
            else:
                a, fa = c, fc

        print(f"Maximo de iteraciones alcanzado")
        return c

    # ==================== GAUSS-SEIDEL ====================
    def gauss_seidel(self, A, b, x0=None, tol=1e-6, max_iter=100):
        print("\n" + "=" * 60)
        print("METODO DE GAUSS-SEIDEL")
        print("=" * 60)

        n = len(b)
        x = np.zeros(n) if x0 is None else x0.copy()

        print("\nMatriz A:\n", A)
        print("Vector b:", b)

        # Verificar diagonal dominante
        print("\nDiagonal Dominante:")
        for i in range(n):
            suma = sum(abs(A[i, j]) for j in range(n) if j != i)
            check = "OK" if abs(A[i, i]) >= suma else "X"
            print(f"Fila {i + 1}: |{A[i, i]:.2f}| >= {suma:.2f} {check}")

        print("\n{:<5} {:<50} {:<15}".format("Iter", "x", "Error"))
        print("-" * 70)

        for k in range(max_iter):
            x_old = x.copy()

            for i in range(n):
                suma = sum(A[i, j] * x[j] for j in range(n) if j != i)
                x[i] = (b[i] - suma) / A[i, i]

            error = np.linalg.norm(x - x_old, np.inf)
            x_str = np.array2string(x, precision=6, suppress_small=True)
            print(f"{k + 1:<5} {x_str:<50} {error:<15.2e}")

            if error < tol:
                print(f"\nSolucion: x = {x}")
                print(f"Verificacion ||Ax - b||: {np.linalg.norm(A @ x - b):.2e}")
                return x

        print("Maximo de iteraciones alcanzado")
        return x

    # ==================== LAGRANGE ====================
    def lagrange(self, x_data, y_data, x_eval=None):
        print("\n" + "=" * 60)
        print("INTERPOLACION DE LAGRANGE")
        print("=" * 60)

        n = len(x_data)
        print(f"\nPuntos ({n}):")
        for i in range(n):
            print(f"  ({x_data[i]:.6f}, {y_data[i]:.6f})")

        x = sp.Symbol('x')
        polinomio = 0

        for i in range(n):
            Li = 1
            for j in range(n):
                if i != j:
                    Li *= (x - x_data[j]) / (x_data[i] - x_data[j])
            polinomio += y_data[i] * Li

        polinomio = sp.expand(polinomio)
        print(f"\nP(x) = {polinomio}")

        if x_eval is not None:
            func = sp.lambdify(x, polinomio, 'numpy')
            y_eval = func(x_eval)
            print(f"\nP({x_eval}) = {y_eval:.10f}")

        return polinomio

    # ==================== SIMPSON ====================
    def simpson(self, func_str, a, b, n=100):
        print("\n" + "=" * 60)
        print("REGLA DE SIMPSON (1/3)")
        print("=" * 60)

        if n % 2 != 0:
            n += 1

        func_sym = sp.sympify(func_str)
        func = sp.lambdify(self.x, func_sym, 'numpy')

        print(f"Funcion: f(x) = {func_str}")
        print(f"Limites: [{a}, {b}] | Subintervalos: n={n}")

        h = (b - a) / n
        x_vals = np.linspace(a, b, n + 1)
        y_vals = func(x_vals)

        integral = y_vals[0] + y_vals[-1]
        integral += 4 * sum(y_vals[i] for i in range(1, n, 2))
        integral += 2 * sum(y_vals[i] for i in range(2, n, 2))
        integral *= h / 3

        print(f"\nResultado: Integral [{a},{b}] f(x)dx = {integral:.10f}")

        # Valor exacto si es posible
        try:
            exacta = float(sp.integrate(func_sym, (self.x, a, b)))
            error = abs(exacta - integral)
            print(f"Valor exacto: {exacta:.10f}")
            print(f"Error: {error:.2e} ({error / abs(exacta) * 100:.4f}%)")
        except:
            pass

        return integral

    # ==================== DERIVADA ====================
    def derivada_segunda_orden(self, func_str, x0, h=0.01):
        print("\n" + "=" * 60)
        print("DERIVADA NUMERICA (2DO ORDEN)")
        print("=" * 60)

        func_sym = sp.sympify(func_str)
        func = sp.lambdify(self.x, func_sym, 'numpy')

        print(f"Funcion: f(x) = {func_str}")
        print(f"Punto: x0 = {x0} | Paso: h = {h}")
        print("Formula: f'(x0) = [f(x0+h) - f(x0-h)] / (2h)")

        f_mas = func(x0 + h)
        f_menos = func(x0 - h)
        derivada = (f_mas - f_menos) / (2 * h)

        print(f"\nf(x0+h) = {f_mas:.10f}")
        print(f"f(x0-h) = {f_menos:.10f}")
        print(f"\nf'({x0}) = {derivada:.10f}")

        # Derivada exacta
        try:
            der_sym = sp.diff(func_sym, self.x)
            der_exacta = float(sp.lambdify(self.x, der_sym, 'numpy')(x0))
            error = abs(der_exacta - derivada)
            print(f"Derivada analitica: f'(x) = {der_sym}")
            print(f"f'({x0}) exacto = {der_exacta:.10f}")
            print(f"Error: {error:.2e} ({error / abs(der_exacta) * 100:.4f}%)")
        except:
            pass

        return derivada

    # ==================== EULER ====================
    def euler(self, func_str, t0, y0, tf, h):
        print("\n" + "=" * 60)
        print("METODO DE EULER")
        print("=" * 60)

        func_sym = sp.sympify(func_str)
        func = sp.lambdify((self.t, self.y), func_sym, 'numpy')

        print(f"EDO: dy/dt = {func_str}")
        print(f"Condicion: y({t0}) = {y0}")
        print(f"Intervalo: [{t0}, {tf}] | Paso: h={h}")

        n_pasos = int((tf - t0) / h)
        print(f"\n{0:<5} {t0:<15.6f} {y0:<15.6f}")

        t, y = t0, y0

        for i in range(n_pasos):
            y = y + h * func(t, y)
            t = t + h
            print(f"{i + 1:<5} {t:<15.6f} {y:<15.6f}")

        print(f"\ny({tf}) = {y:.10f}")
        return t, y

    # ==================== RUNGE-KUTTA 4 SISTEMA ====================
    def runge_kutta_4_sistema(self, funcs_str, t0, y0, tf, h):
        print("\n" + "=" * 60)
        print("RUNGE-KUTTA 4TO ORDEN - SISTEMA")
        print("=" * 60)

        n_eq = len(funcs_str)
        y_symbols = [sp.Symbol(f'y{i}') for i in range(n_eq)]

        print(f"Sistema de {n_eq} ecuaciones:")
        for i, func_str in enumerate(funcs_str):
            print(f"  dy{i}/dt = {func_str}")

        print(f"Condicion inicial: y({t0}) = {y0}")
        print(f"Intervalo: [{t0}, {tf}] | Paso: h={h}")

        funcs = []
        for func_str in funcs_str:
            func_sym = sp.sympify(func_str)
            func = sp.lambdify([self.t] + y_symbols, func_sym, 'numpy')
            funcs.append(func)

        n_pasos = int((tf - t0) / h)
        t, y = t0, y0.copy()

        print(f"\n{0:<5} {t:<15.6f} {str(y)}")

        for i in range(n_pasos):
            k1 = np.array([f(t, *y) for f in funcs])
            k2 = np.array([f(t + 0.5 * h, *(y + 0.5 * h * k1)) for f in funcs])
            k3 = np.array([f(t + 0.5 * h, *(y + 0.5 * h * k2)) for f in funcs])
            k4 = np.array([f(t + h, *(y + h * k3)) for f in funcs])

            y = y + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            t = t + h

            print(f"{i + 1:<5} {t:<15.6f} {str(y)}")

        print(f"\nSolucion final en t={tf}:")
        for i, val in enumerate(y):
            print(f"  y{i}({tf}) = {val:.10f}")

        return t, y


def mostrar_menu():
    print("\n1. Biseccion")
    print("2. Gauss-Seidel")
    print("3. Lagrange")
    print("4. Simpson")
    print("5. Derivada Numerica")
    print("6. Euler")
    print("7. Sistema EDO (RK4)")
    print("0. Salir")
    print("-" * 70)


def main():
    metodos = MetodosNumericos()

    while True:
        mostrar_menu()

        try:
            opcion = input("\nOpcion: ").strip()

            if opcion == '0':
                print("\nHasta luego!")
                break

            elif opcion == '1':
                func_str = input("\nf(x): ")
                a = float(input("a: "))
                b = float(input("b: "))
                tol = float(input("Tolerancia (Enter=1e-6): ") or "1e-6")
                max_iter = int(input("Max iteraciones (Enter=100): ") or "100")
                metodos.biseccion(func_str, a, b, tol, max_iter)

            elif opcion == '2':
                n = int(input("\nTamano del sistema: "))
                print("Matriz A (fila por fila):")
                A = np.array([[float(x) for x in input(f"Fila {i + 1}: ").split()] for i in range(n)])
                b = np.array([float(x) for x in input("Vector b: ").split()])

                if input("Vector inicial? (s/n): ").lower() == 's':
                    x0 = np.array([float(x) for x in input("x0: ").split()])
                else:
                    x0 = None

                tol = float(input("Tolerancia (Enter=1e-6): ") or "1e-6")
                max_iter = int(input("Max iteraciones (Enter=100): ") or "100")
                metodos.gauss_seidel(A, b, x0, tol, max_iter)

            elif opcion == '3':
                n = int(input("\nNumero de puntos: "))
                x_data = [float(input(f"x_{i}: ")) for i in range(n)]
                y_data = [float(input(f"y_{i}: ")) for i in range(n)]

                x_eval = None
                if input("Evaluar? (s/n): ").lower() == 's':
                    x_eval = float(input("x: "))

                metodos.lagrange(np.array(x_data), np.array(y_data), x_eval)

            elif opcion == '4':
                func_str = input("\nf(x): ")
                a = float(input("a: "))
                b = float(input("b: "))
                n = int(input("Subintervalos (Enter=100): ") or "100")
                metodos.simpson(func_str, a, b, n)

            elif opcion == '5':
                func_str = input("\nf(x): ")
                x0 = float(input("x0: "))
                h = float(input("h (Enter=0.01): ") or "0.01")
                metodos.derivada_segunda_orden(func_str, x0, h)

            elif opcion == '6':
                func_str = input("\nf(t,y): ")
                t0 = float(input("t0: "))
                y0 = float(input("y0: "))
                tf = float(input("tf: "))
                h = float(input("h: "))
                metodos.euler(func_str, t0, y0, tf, h)

            elif opcion == '7':
                n_eq = int(input("\nNumero de ecuaciones: "))
                funcs_str = [input(f"dy{i}/dt = ") for i in range(n_eq)]
                t0 = float(input("t0: "))
                y0 = np.array([float(input(f"y{i}({t0}) = ")) for i in range(n_eq)])
                tf = float(input("tf: "))
                h = float(input("h: "))
                metodos.runge_kutta_4_sistema(funcs_str, t0, y0, tf, h)

            else:
                print("Opcion invalida")

            input("\n[Enter para continuar]")

        except KeyboardInterrupt:
            print("\n\nInterrumpido.")
            break
        except Exception as e:
            print(f"\nERROR: {e}")
            input("\n[Enter para continuar]")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  ANALISIS NUMERICO PARA PROGRAMACION")
    print("  PROGRAMA DE METODOS NUMERICOS")
    print("  Alfredo S - 2095320 - Semestre 5 - FCFM")
    print("=" * 70)
    main()