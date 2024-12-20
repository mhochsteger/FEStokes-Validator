from webapp_client.app import App
from webapp_client.components import *
from webapp_client.qcomponents import *
from webapp_client.visualization import SolutionWebgui
from webapp_client.utils import load_image
import netgen.occ as ngocc
import ngsolve as ngs
import os


def image(filename):
    picture = os.path.join(os.path.dirname(__file__), "assets", filename)
    return load_image(picture)


mesh_cards = {
    "Unstructured Mesh": {"image": "mesh/stdmesh.webp"},
    "Curved Mesh": {"image": "mesh/curvedmesh.webp"},
    "Type One Mesh": {"image": "mesh/typeonemesh.webp"},
    "Singular Vertex Mesh": {"image": "mesh/crisscross.webp"},
    "None": {"image": "mesh/emptymesh.webp"},
}

pressure_cards = {
    "P0": {"image": "pressure/Pzeropressure.webp"},
    "P1": {"image": "pressure/Ponepressure.webp"},
    "P1*": {"image": "pressure/Ponedpressure.webp"},
    "P2": {"image": "pressure/Ptwopressure.webp"},
    "P2*": {"image": "pressure/Ptwodpressure.webp"},
    "P3": {"image": "pressure/Pthreepressure.webp"},
    "P3*": {"image": "pressure/Pthreedpressure.webp"},
    "None": {"image": "pressure/emptypressure.webp"},
}

velocity_cards = {
    "P1": {"image": "velocity/Ponevel.webp"},
    "P1*": {"image": "velocity/Ponedvel.webp"},
    "BDM1": {"image": "velocity/BDMonevel.webp"},
    "Crouzeix-Raviart": {"image": "velocity/CRvel.webp"},
    "P2": {"image": "velocity/Ptwovel.webp"},
    "P2*": {"image": "velocity/Ptwodvel.webp"},
    "BDM2": {"image": "velocity/BDMtwovel.webp"},
    "P3": {"image": "velocity/Pthreevel.webp"},
    "P3*": {"image": "velocity/Pthreedvel.webp"},
    "BDM3": {"image": "velocity/BDMthreevel.webp"},
    "BDM4": {"image": "velocity/BDMfourvel.webp"},
    "P4": {"image": "velocity/Pfourvel.webp"},
    "P4*": {"image": "velocity/Pfourdvel.webp"},
    "None": {"image": "velocity/emptyvel.webp"},
}

extra_cards = {
    "Interior Penalty": {"image": "extra/ipdg.webp"},
    "Pressure-Jump": {"image": "extra/pj.webp"},
    "Powell-Sabin Split": {"image": "extra/psmesh.webp"},
    "Alfeld Split": {"image": "extra/alfeldsplit.webp"},
    "Brezzi-Pitkäranta": {"image": "extra/bp.webp"},
    "P3 Bubble": {"image": "extra/Pthreebubble.webp"},
    "None": {"image": "extra/emptyextra.webp"},
}


class CardSelector(QCard):
    def __init__(self, options, label):
        self._options = options
        self.selector = QSelect(
            options=list(options.keys()), model_value="None", label=label
        )
        self.selector.on_update_model_value(self.update)
        self.div_image = QImg(
            src=image(options[self.selector.model_value]["image"]), width="200px"
        )
        super().__init__(
            self.selector, self.div_image, style="padding: 10px; margin: 10px;"
        )

    def update(self):
        print("selected item =", self.selector.model_value)
        self.div_image.src = image(self._options[self.selector.model_value]["image"])

    def on_update_model_value(self, callback):
        self.selector.on_update_model_value(callback)

    @property
    def model_value(self):
        return self.selector.model_value

    @model_value.setter
    def model_value(self, value):
        self.selector.model_value = value


class FeStokesRePair(App):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mesh = CardSelector(
            label="Mesh",
            options=mesh_cards,
        )

        # self.mesh.on_update_model_value(self.calculate)
        self.pressure = CardSelector(
            label="Pressure",
            options=pressure_cards,
        )
        # self.pressure.on_update_model_value(self.calculate)
        self.velocity = CardSelector(
            label="Velocity",
            options=velocity_cards,
        )
        # self.velocity.on_update_model_value(self.calculate)
        self.add_extra = Row(
            QBtn(round=True, icon="add", fab=True).on_click(self._add_extra),
            classes="items-center",
        )
        self.clear_btn = QBtn(label="Clear").on_click(self.clear)
        self.calc_btn = QBtn(label="Validate").on_click(self.calculate)
        self.extras = Row()
        self.velocity_sol = SolutionWebgui(
            caption="Velocity", show_clipping=False, show_view=False
        )
        self.pressure_sol = SolutionWebgui(
            caption="Pressure", show_clipping=False, show_view=False
        )
        self.user_warning = UserWarning(
            title="Error in calculation!", message="Pairing does not seem to work"
        )

        self.cards = Row(
            self.mesh, self.pressure, self.velocity, self.extras, self.add_extra
        )
        self.computing = QInnerLoading(
            QSpinnerGears(size="100px", color="primary"),
            Centered("Calculating..."),
            showing=True,
            style="z-index:100;",
        )
        self.computing.hidden = True

        self.result_section = Row(
            self.computing,
            Col(Heading("Velocity", level=3), self.velocity_sol),
            Col(Heading("Pressure", level=3), self.pressure_sol),
        )
        self.component = Centered(
            Col(
                self.user_warning,
                self.cards,
                Row(self.clear_btn, self.calc_btn),
                self.result_section,
                classes="q-gutter-lg q-ma-lg",
            )
        )

    def clear(self):
        self.extras.children = []
        self.mesh.model_value = "None"
        self.mesh.update()
        self.pressure.model_value = "None"
        self.pressure.update()
        self.velocity.model_value = "None"
        self.velocity.update()
        self.velocity_sol._webgui.clear()
        self.pressure_sol._webgui.clear()

    def _add_extra(self):
        i = len(self.extras.children)
        extra = CardSelector(
            label="Extra " + str(i + 1),
            options=extra_cards,
        )
        # extra.on_update_model_value(self.calculate)
        self.extras.children = self.extras.children + [extra]

    def calculate(self):
        if self.mesh.model_value is None:
            return
        self.computing.hidden = False
        mesh = self._create_mesh()
        if self.velocity.model_value is None or self.pressure.model_value is None:
            self.velocity_sol.draw(mesh)
            self.pressure_sol.draw(mesh)
            self.computing.hidden = True
            return
        try:
            self._solve_stokes(mesh)
        except Exception as e:
            print("caught exception", e)
            self.user_warning.message = str(e)
            self.user_warning.show()
            self.velocity_sol._webgui.clear()
            self.pressure_sol._webgui.clear()
        self.computing.hidden = True

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
        bubble_space = False
        if "P3 Bubble" in [e.model_value for e in self.extras.children]:
            bubble_space = True
            print("Add P3 Bubble")
            Vhs = ngs.VectorH1(mesh, order=3)
            bubbles = ngs.BitArray(Vhs.ndof)
            bubbles.Clear()
            for dof in range(
                mesh.nv + 2 * mesh.nedge, mesh.nv + 2 * mesh.nedge + mesh.ne
            ):
                bubbles.Set(dof)
                bubbles.Set(Vhs.ndof // 2 + dof)
            Vhb = ngs.Compress(Vhs, active_dofs=bubbles)
            V *= Vhb
        print("Create Pressure space")
        if self.pressure.model_value.endswith("*"):
            print(f"Create L2({int(self.pressure.model_value[1])})")
            Q = ngs.L2(mesh, order=int(self.pressure.model_value[1]))
        else:
            print(f"Create H1({int(self.pressure.model_value[1])})")
            Q = ngs.H1(mesh, order=int(self.pressure.model_value[1]))
        fes = V * Q
        if bubble_space:
            print("in bubble space")
            (u, ub, p), (v, vb, q) = fes.TnT()
            gradu = ngs.Grad(u) + ngs.Grad(ub)
            gradv = ngs.Grad(v) + ngs.Grad(vb)
            divu = ngs.div(u) + ngs.div(ub)
            divv = ngs.div(v) + ngs.div(vb)
        else:
            (u, p), (v, q) = fes.TnT()
            gradu, gradv = ngs.Grad(u), ngs.Grad(v)
            divu, divv = ngs.div(u), ngs.div(v)

        stokes = (
            ngs.InnerProduct(gradu, gradv) * ngs.dx
            + divu * q * ngs.dx
            + divv * p * ngs.dx
        )
        a = ngs.BilinearForm(stokes).Assemble()
        gf = ngs.GridFunction(fes)
        if bubble_space:
            gfu, gfb, gfp = gf.components
            vel = gfu + gfb
        else:
            gfu, gfp = gf.components
            vel = gfu
        uin = ngs.CF((1.5 * 4 * ngs.y * (0.41 - ngs.y) / (0.41 * 0.41), 0))
        gfu.Set(uin, definedon=mesh.Boundaries("left"))
        res = (-a.mat * gf.vec).Evaluate()
        inv = ngs.directsolvers.SuperLU(a.mat, fes.FreeDofs())
        gf.vec.data += inv * res
        self.velocity_sol.draw(vel, mesh)
        self.pressure_sol.draw(gfp, mesh)
