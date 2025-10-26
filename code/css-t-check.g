LoadPackage("guava","0");;

G1:=[
[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
[1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
[1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0],
[1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0],
[1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0]
];

G2:=[
[1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
[1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0],
[1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0],
[1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0]
];

C1 := GeneratorMatCode(G1,GF(2));
C2 := GeneratorMatCode(G2,GF(2));

D1 := DualCode(C1);
D2 := DualCode(C2);

HullC1 := Intersection(C1, D1);
HullC2 := Intersection(C2, D2);

cond1 := IsSubset(HullC1, C2) and IsSubset(HullC2, C2);
Print("C2 ⊆ Hull(C1) ∩ Hull(C2)? ", cond1, "\n");

HullCond := Intersection(C2, D1);
cond2 := IsSubset(HullCond, C2);
Print("C2 ⊆ Hull_{C1}(C2)? ", cond2, "\n");

if cond1 = cond2 then
    Print("Equivalent conditions satisfied: CSS-T condition holds.\n");
else
    Print("The CSS-T condition does NOT hold.\n");
fi;