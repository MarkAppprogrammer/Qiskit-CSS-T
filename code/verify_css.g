LoadPackage("guava","0");;

G2:=[
[1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
[1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0],
[1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0],
[1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0]
];

Read("parity_mats.gap");

H1 := Hs[1] * Z(2);
H2 := Hs[2] * Z(2);

Hx := H1;
Hz := G2 * Z(2);

S := Hx * TransposedMat(Hz);

Print("HX * HZ^T =\n"); Display(S);
Print("\nOrthogonality condition holds?  ", S = 0*S, "\n");

filepath := "Qiskit-CSS-T/matrices/parity_maps_xz.gap";
file := OutputTextFile(filepath, false);
PrintTo(file, "Hx := ", Hx, ";\n");
AppendTo(file, "Hz := ", Hz, ";\n");
CloseStream(file);

Print("\nSaved Hx and Hz to: ", filepath, "\n");
