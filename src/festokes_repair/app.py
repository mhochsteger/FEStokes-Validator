from webapp_client.app import App
from webapp_client.components import *
from webapp_client.qcomponents import *
from webapp_client.visualization import SolutionWebgui
from webapp_client.utils import compute_node
import netgen.occ as ngocc
import ngsolve as ngs


class FeStokesRePair(App):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.progress = QLinearProgress(value=0.0, animation_speed=100)
        self.mesh = QSelect(
            label="Mesh",
            id="mesh",
            options=[
                "Unstructured Mesh",
                "Curved Mesh",
                "Type One Mesh",
                "Singular Vertex Mesh",
            ],
        )
        self.mesh.on_update_model_value(self.calculate)
        self.pressure = QSelect(
            label="Pressure",
            id="pressure",
            options=["P0", "P1", "P1*", "P2", "P2*", "P3", "P3*"],
        )
        self.pressure.on_update_model_value(self.calculate)
        self.velocity = QSelect(
            label="Velocity",
            id="velocity",
            options=[
                "P0",
                "P1",
                "P1*",
                "BDM1",
                "Crouzeix-Raviart",
                "P2",
                "P2*",
                "BDM2",
                "P3",
                "P3*",
            ],
        )
        self.velocity.on_update_model_value(self.calculate)
        self.add_extra = QBtn(label="Add Extra").on_click(self._add_extra)
        self.clear_btn = QBtn(label="Clear").on_click(self.clear)
        self.extras = Div()
        self.velocity_sol = SolutionWebgui(
            caption="Velocity", id="velocity_sol", show_clipping=False, show_view=False
        )
        self.pressure_sol = SolutionWebgui(
            caption="Pressure", id="pressure_sol", show_clipping=False, show_view=False
        )
        self.user_warning = UserWarning(
            title="Error in calculation!", message="Pairing does not seem to work"
        )

        self.component = Centered(
            Col(
                self.user_warning,
                self.mesh,
                self.pressure,
                self.velocity,
                self.extras,
                Row(self.add_extra, self.clear_btn),
                Row(
                    Col(Heading("Velocity", level=3), self.velocity_sol),
                    Col(Heading("Pressure", level=3), self.pressure_sol),
                ),
                classes="q-gutter-lg q-ma-lg",
            )
        )

    def clear(self):
        self.extras.children = []
        self.mesh.model_value = None
        self.pressure.model_value = None
        self.velocity.model_value = None
        self.velocity_sol._webgui.clear()
        self.pressure_sol._webgui.clear()

    def _add_extra(self):
        i = len(self.extras.children)
        extra = QSelect(
            label="Extra " + str(i + 1),
            id=f"extra_{i}",
            options=[
                "Interior Penalty",
                "Pressure-Jump",
                "Powell-Sabin Split",
                "Alfeld Split",
                "Brezzi-Pitkäranta",
                "P3 Bubble",
            ],
        )
        extra.on_update_model_value(self.calculate)
        self.extras.children = self.extras.children + [extra]

    def calculate(self):
        if self.mesh.model_value is None:
            return
        mesh = self._create_mesh()
        if self.velocity.model_value is None or self.pressure.model_value is None:
            self.velocity_sol.draw(mesh)
            self.pressure_sol.draw(mesh)
            return
        try:
            self._solve_stokes(mesh)
        except Exception as e:
            print("caught exception", e)
            self.user_warning.message = str(e)
            self.user_warning.show()
            self.velocity_sol._webgui.clear()
            self.pressure_sol._webgui.clear()

    def _create_mesh(self):
        import ngsolve.meshes as ngs_meshes

        print("Create mesh")
        if self.mesh.model_value in ["Unstructured Mesh", "Curved Mesh"]:
            shape = ngocc.Rectangle(2, 0.41).Circle(0.2, 0.2, 0.05).Reverse().Face()
            shape.edges.name = "top"
            shape.edges.Min(ngocc.X).name = "left"
            shape.edges.Max(ngocc.X).name = "right"
            geo = ngocc.OCCGeometry(shape, dim=2)
            mesh = ngs.Mesh(geo.GenerateMesh(maxh=0.05))
        elif self.mesh.model_value == "Type One Mesh":
            mesh = ngs_meshes.MakeStructured2DMesh(quads=False, nx=10, ny=10)
        else:  # self.mesh.model_value == "Singular Vertex Mesh":
            mesh = ngs_meshes.MakeStructured2DMesh(quads=True, nx=10, ny=10)
            # split quads in 4 trigs?
        for e in self.extras.children:
            if e.model_value == "Alfeld Split":
                ngmesh = mesh.ngmesh
                ngmesh.SplitAlfeld()
                mesh = ngs.Mesh(ngmesh)
            elif e.model_value == "Powell-Sabin Split":
                ngmesh = mesh.ngmesh
                ngmesh.SplitPowellSabin()
                mesh = ngs.Mesh(ngmesh)
        if self.mesh.model_value == "Curved Mesh":
            mesh.Curve(5)
        return mesh

    def _solve_stokes(self, mesh):
        assert self.velocity.model_value is not None
        assert self.pressure.model_value is not None
        print("Create Velocity space")
        if self.velocity.model_value == "Crouzeix-Raviart":
            print("Create Crouzeix-Raviart")
            V = ngs.FESpace("nonconforming", mesh, order=1, dirichlet="top|left") ** 2
        elif self.velocity.model_value.startswith("BDM"):
            print("Create BDM of order", self.velocity.model_value[-1])
            V = ngs.HDiv(mesh, order=int(self.velocity.model_value[-1]))
        else:
            order = int(self.velocity.model_value[1])
            print("Create P", order)
            V = ngs.VectorH1(mesh, order=order, dirichlet="top|left")
            if self.velocity.model_value.endswith("*"):
                print("Make discontinuous")
                V = ngs.Discontinuous(V)
        print("Create Pressure space")
        if self.pressure.model_value.endswith("*"):
            print(f"Create L2({int(self.pressure.model_value[1])})")
            Q = ngs.L2(mesh, order=int(self.pressure.model_value[1]))
        else:
            print(f"Create H1({int(self.pressure.model_value[1])})")
            Q = ngs.H1(mesh, order=int(self.pressure.model_value[1]))
        fes = V * Q
        (u, p), (v, q) = fes.TnT()
        stokes = (
            ngs.InnerProduct(ngs.Grad(u), ngs.Grad(v)) * ngs.dx
            + ngs.div(u) * q * ngs.dx
            + ngs.div(v) * p * ngs.dx
        )
        a = ngs.BilinearForm(stokes).Assemble()
        gf = ngs.GridFunction(fes)
        gfu, gfp = gf.components
        uin = ngs.CF((1.5 * 4 * ngs.y * (0.41 - ngs.y) / (0.41 * 0.41), 0))
        gfu.Set(uin, definedon=mesh.Boundaries("left"))
        res = (-a.mat * gf.vec).Evaluate()
        inv = ngs.directsolvers.SuperLU(a.mat, fes.FreeDofs())
        gf.vec.data += inv * res
        self.velocity_sol.draw(gfu, mesh)
        self.pressure_sol.draw(gfp, mesh)
