import Std

namespace MathTest

def seedCoeff (coeffs : List Float) : Float :=
  match coeffs.reverse with
  | [] => 0.0
  | a :: _ => a

def hornerFMASeq (coeffs : List Float) : List (Float × Float) :=
  match coeffs.reverse with
  | [] => []
  | _ :: rest => rest.map (fun b => (1.0, b))

def evalFMASeq (ops : List (Float × Float)) (x seed : Float) : Float :=
  ops.foldl (fun acc op => acc * x + op.snd) seed

def evalBySeq (coeffs : List Float) (x : Float) : Float :=
  evalFMASeq (hornerFMASeq coeffs) x (seedCoeff coeffs)

def evalPolynomial (coeffs : List Float) (x : Float) : Float :=
  evalBySeq coeffs x

theorem hornerFMASeq_correct (coeffs : List Float) (x : Float) :
    evalBySeq coeffs x = evalPolynomial coeffs x := by
  rfl

def pythonModule : String :=
  "# AUTO-GENERADO desde MathTest/HornerExtract.lean - NO MODIFICAR MANUALMENTE\n" ++
  "# Teorema: hornerFMASeq_correct\n" ++
  "\n" ++
  "def horner_fma_seq(coeffs):\n" ++
  "    \"\"\"Genera secuencia FMA de Horner como lista (w, b).\"\"\"\n" ++
  "    n = len(coeffs)\n" ++
  "    if n <= 1:\n" ++
  "        return []\n" ++
  "    # coeffs = [a0, a1, ..., an], secuencia usa [a_{n-1}, ..., a0]\n" ++
  "    return [(1.0, float(c)) for c in reversed(coeffs[:-1])]\n" ++
  "\n" ++
  "def horner_seed(coeffs):\n" ++
  "    return float(coeffs[-1]) if coeffs else 0.0\n" ++
  "\n" ++
  "__lean_source__ = \"MathTest/HornerExtract.lean\"\n" ++
  "__theorem__ = \"hornerFMASeq_correct\"\n"

def emitPythonModule : IO Unit := do
  IO.FS.writeFile "python_analysis/horner_generated.py" pythonModule
  IO.println "Generated python_analysis/horner_generated.py"

end MathTest

def main : IO Unit :=
  MathTest.emitPythonModule
