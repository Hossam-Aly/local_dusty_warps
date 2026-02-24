#include "idefix.hpp"
#include "setup.hpp"

static real omega;
static real shear;
static real psi;
static real cs;
real epsilon;
real chi;
real tauGlob;
real pi=3.14;


#define  FILENAME    "timevol.dat"

//#define STRATIFIED
void PressureGradient(Hydro *hydro, const real t, const real dt) {
  auto Uc = hydro->Uc;
  auto Vc = hydro->Vc;
  DataBlock *data = hydro->data;
  real eps = epsilon;
  idefix_for("MySourceTerm",0,data->np_tot[KDIR],0,data->np_tot[JDIR],0,data->np_tot[IDIR],
              KOKKOS_LAMBDA (int k, int j, int i) {
                // Radial pressure gradient
                  Uc(MX1,k,j,i) += eps*Vc(RHO,k,j,i)*dt;
              });
}

void BodyForce(DataBlock &data, const real t, IdefixArray4D<real> &force) {
  idfx::pushRegion("BodyForce");
  IdefixArray1D<real> x = data.x[IDIR];
  IdefixArray1D<real> z = data.x[KDIR];

  // GPUS cannot capture static variables
  real omegaLocal=omega;
  real shearLocal =shear;

  idefix_for("BodyForce",
              data.beg[KDIR] , data.end[KDIR],
              data.beg[JDIR] , data.end[JDIR],
              data.beg[IDIR] , data.end[IDIR],
              KOKKOS_LAMBDA (int k, int j, int i) {

                force(IDIR,k,j,i) = -2.0*omegaLocal*shearLocal*x(i);
                force(JDIR,k,j,i) = ZERO_F;
                force(KDIR,k,j,i) = ZERO_F;

      });


  idfx::popRegion();
}

void InducedWarpFlow(Hydro *hydro, const real t, const real dt) {
  auto Uc = hydro->Uc;
  auto Vc = hydro->Vc;
  DataBlock *data = hydro->data;
  IdefixArray1D<real> z = data->x[KDIR];

  real omegaLocal=omega;
  real psiLocal = psi;

  idefix_for("MySourceTerm",0,data->np_tot[KDIR],0,data->np_tot[JDIR],0,data->np_tot[IDIR],
              KOKKOS_LAMBDA (int k, int j, int i) {
                  Uc(MX1,k,j,i) += psiLocal*omegaLocal*omegaLocal*z(k)*sin(omegaLocal*t)*dt;
              });
}

void InducedWarp(DataBlock &data, const real t, IdefixArray4D<real> &force) {
  idfx::pushRegion("InducedWarp");
  IdefixArray1D<real> z = data.x[KDIR];

  // GPUS cannot capture static variables
  real omegaLocal=omega;
  real psiLocal = psi;

  idefix_for("InducedWarp",
              data.beg[KDIR] , data.end[KDIR],
              data.beg[JDIR] , data.end[JDIR],
              data.beg[IDIR] , data.end[IDIR],
              KOKKOS_LAMBDA (int k, int j, int i) {

                force(IDIR,k,j,i) = psiLocal*omegaLocal*omegaLocal*z(k)*sin(omegaLocal*t);
                force(JDIR,k,j,i) = ZERO_F;
                force(KDIR,k,j,i) = ZERO_F;

      });


  idfx::popRegion();

}

void nonReflective(DataBlock& data, int dir, BoundarySide side, real t) {
  idfx::pushRegion("nonReflective");
  IdefixArray4D<real> Vc = data.hydro->Vc;
  IdefixArray4D<real> Uc = data.dust[0]->Vc;

  IdefixArray1D<real> x = data.x[IDIR];
  IdefixArray1D<real> z = data.x[KDIR];

  real shearLocal = shear;
  real chilocal = chi;


  int nxi = data.np_int[IDIR];
  int nxj = data.np_int[JDIR];
  int nxk = data.np_int[KDIR];

  const int ighost = data.nghost[IDIR];
  const int jghost = data.nghost[JDIR];
  const int kghost = data.nghost[KDIR];
  //DataBlockHost d(data);

  
  data.hydro->boundary->BoundaryFor("nonReflective", dir, side,
    KOKKOS_LAMBDA (int k, int j, int i) {
      int iref, jref, kref;
        // This hack takes care of cases where we have more ghost zones than active zones
        //real x=d.x[IDIR](i);
        if(dir==IDIR)
          iref = ighost + (i+ighost*(nxi-1))%nxi;
        else
          iref = i;
        if(dir==JDIR)
          jref = jghost + (j+jghost*(nxj-1))%nxj;
        else
          jref = j;
        if(dir==KDIR)
          kref = kghost + (k+kghost*(nxk-1))%nxk;
        else
          kref = k;
        
        Vc(RHO,k,j,i) = Vc(RHO,kref,jref,iref);
        Vc(VX3,k,j,i) = Vc(VX3,kref,jref,iref);
        if(kref == k){
          Vc(VX1,k,j,i) = Vc(VX1,kref,jref,iref);
          Vc(VX2,k,j,i) = Vc(VX2,kref,jref,iref);
        }
        else{
          Vc(VX1,k,j,i) = 4.142938*z(k)*cos(t);
          Vc(VX2,k,j,i) = -2.1162*z(k)*sin(t)+shearLocal*x(i);
        }
        });
  data.dust[0]->boundary->BoundaryFor("nonReflective", dir, side,
    KOKKOS_LAMBDA (int k, int j, int i) {
      int iref, jref, kref;
        // This hack takes care of cases where we have more ghost zones than active zones
        //real x=d.x[IDIR](i);
        if(dir==IDIR)
          iref = ighost + (i+ighost*(nxi-1))%nxi;
        else
          iref = i;
        if(dir==JDIR)
          jref = jghost + (j+jghost*(nxj-1))%nxj;
        else
          jref = j;
        if(dir==KDIR)
          kref = kghost + (k+kghost*(nxk-1))%nxk;
        else
          kref = k;
        Uc(RHO,k,j,i) = Uc(RHO,kref,jref,iref);
        Uc(VX3,k,j,i) = Uc(VX3,kref,jref,iref);
        if(kref == k){
          Uc(VX1,k,j,i) = Uc(VX1,kref,jref,iref);
          Uc(VX2,k,j,i) = Uc(VX2,kref,jref,iref);
        }
        else{
          Uc(VX1,k,j,i) = 4.160947*z(k)*cos(t);
          Uc(VX2,k,j,i) = -2.1067767*z(k)*sin(t)+shearLocal*x(i);
        }


        });
}

void ApplyBoundaryReversePeriodic(DataBlock *data, IdefixArray4D<real> Vc, int dir, BoundarySide side) {
  idfx::pushRegion("reversePeriodic");
  IdefixArray1D<real> x = data->x[IDIR];
  real shearLocal = shear;
  int nxi = data->np_int[IDIR];
  int nxj = data->np_int[JDIR];
  int nxk = data->np_int[KDIR];
  const int ighost = data->nghost[IDIR];
  const int jghost = data->nghost[JDIR];
  const int kghost = data->nghost[KDIR];

    data->hydro->boundary->BoundaryFor("reversePeriodic", dir, side,
      KOKKOS_LAMBDA (int k, int j, int i) {
        int iref, jref, kref;
        // This hack takes care of cases where we have more ghost zones than active zones
        //real x=d.x[IDIR](i);
        if(dir==IDIR)
          iref = ighost + (i+ighost*(nxi-1))%nxi;
        else
          iref = i;
        if(dir==JDIR)
          jref = jghost + (j+jghost*(nxj-1))%nxj;
        else
          jref = j;
        if(dir==KDIR)
          kref = kghost + (k+kghost*(nxk-1))%nxk;
        else
          kref = k;
        
        Vc(RHO,k,j,i) = Vc(RHO,kref,jref,iref);
        Vc(VX3,k,j,i) = Vc(VX3,kref,jref,iref);
        if(kref == k){
          Vc(VX1,k,j,i) = Vc(VX1,kref,jref,iref);
          Vc(VX2,k,j,i) = Vc(VX2,kref,jref,iref);

        }
        else{
          Vc(VX1,k,j,i) = -Vc(VX1,kref,jref,iref);
          Vc(VX2,k,j,i) = -Vc(VX2,kref,jref,iref)+2.*shearLocal*x(i);
        }
        });
}

void reversePeriodicGas(Hydro *hydro, int dir, BoundarySide side, real t) {
  ApplyBoundaryReversePeriodic(hydro->data, hydro->Vc, dir, side);
}

void reversePeriodicDust(Fluid<DustPhysics> *dust, int dir, BoundarySide side, real t) {
  ApplyBoundaryReversePeriodic(dust->data, dust->Vc, dir, side);
}


void reversePeriodic(DataBlock& data, int dir, BoundarySide side, real t) {
  idfx::pushRegion("reversePeriodic");
  IdefixArray4D<real> Vc = data.hydro->Vc;
  IdefixArray4D<real> Uc = data.dust[0]->Vc;

  IdefixArray1D<real> x = data.x[IDIR];
  real shearLocal = shear;


  int nxi = data.np_int[IDIR];
  int nxj = data.np_int[JDIR];
  int nxk = data.np_int[KDIR];

  const int ighost = data.nghost[IDIR];
  const int jghost = data.nghost[JDIR];
  const int kghost = data.nghost[KDIR];
  //DataBlockHost d(data);

  data.hydro->boundary->BoundaryFor("reversePeriodic", dir, side,
    KOKKOS_LAMBDA (int k, int j, int i) {
      int iref, jref, kref;
        // This hack takes care of cases where we have more ghost zones than active zones
        //real x=d.x[IDIR](i);
        if(dir==IDIR)
          iref = ighost + (i+ighost*(nxi-1))%nxi;
        else
          iref = i;
        if(dir==JDIR)
          jref = jghost + (j+jghost*(nxj-1))%nxj;
        else
          jref = j;
        if(dir==KDIR)
          kref = kghost + (k+kghost*(nxk-1))%nxk;
        else
          kref = k;
        
        Vc(RHO,k,j,i) = Vc(RHO,kref,jref,iref);
        Vc(VX3,k,j,i) = Vc(VX3,kref,jref,iref);
        if(kref == k){
          Vc(VX1,k,j,i) = Vc(VX1,kref,jref,iref);
          Vc(VX2,k,j,i) = Vc(VX2,kref,jref,iref);

        }
        else{
          Vc(VX1,k,j,i) = -Vc(VX1,kref,jref,iref);
          Vc(VX2,k,j,i) = -Vc(VX2,kref,jref,iref)+2.*shearLocal*x(i);
        }
        });
  data.dust[0]->boundary->BoundaryFor("reversePeriodic", dir, side,
    KOKKOS_LAMBDA (int k, int j, int i) {
      int iref, jref, kref;
        // This hack takes care of cases where we have more ghost zones than active zones
        //real x=d.x[IDIR](i);
        if(dir==IDIR)
          iref = ighost + (i+ighost*(nxi-1))%nxi;
        else
          iref = i;
        if(dir==JDIR)
          jref = jghost + (j+jghost*(nxj-1))%nxj;
        else
          jref = j;
        if(dir==KDIR)
          kref = kghost + (k+kghost*(nxk-1))%nxk;
        else
          kref = k;
        
        Uc(RHO,k,j,i) = Uc(RHO,kref,jref,iref);
        Uc(VX3,k,j,i) = Uc(VX3,kref,jref,iref);
        if(kref == k){
          Uc(VX1,k,j,i) = Uc(VX1,kref,jref,iref);
          Uc(VX2,k,j,i) = Uc(VX2,kref,jref,iref);
        }
        else{
          Uc(VX1,k,j,i) = -Uc(VX1,kref,jref,iref);
          Uc(VX2,k,j,i) = -Uc(VX2,kref,jref,iref)+2.*shearLocal*x(i);
        }


        });
}



// Analyse data to produce an output
void Analysis(DataBlock & data) {


// Mirror data on Host
  DataBlockHost d(data);
  // Sync it
  d.SyncFromDevice();
  real rho = d.dustVc[0](RHO,0,0,0);
  real KE = 0.0;
  for(int k = 0; k < d.np_tot[KDIR] ; k++) {
    for(int j = 0; j < d.np_tot[JDIR] ; j++) {
      for(int i = 0; i < d.np_tot[IDIR] ; i++) {
        KE = KE + d.Vc(RHO,k,j,i)*d.Vc(VX3,k,j,i)*d.Vc(VX3,k,j,i);
        if(rho < d.dustVc[0](RHO,k,j,i)) {
          rho=d.dustVc[0](RHO,k,j,i);
        }
      }
    }
  }
  #ifdef WITH_MPI
  real rho_max;
  real KE_total;
  MPI_Reduce(&rho, &rho_max, 1, realMPI, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(&KE, KE_total, 1 realMPI, MPI_SUM, 0, MPI_COMM_WORLD);
  #endif
  if(idfx::prank == 0) {
  std::ofstream f;
  f.open(FILENAME,std::ios::app);
  f.precision(10);
  #ifdef WITH_MPI
  f << std::scientific << data.t << "	" << rho_max << "	" << 0.5/d.np_tot[KDIR]*1.2084/d.np_tot[IDIR]*3.14159/d.np_tot[JDIR]*KE_total << " " << std::endl;
  #else
  f << std::scientific << data.t << "	" << rho << "	" << 0.5/d.np_tot[KDIR]*1.2084/d.np_tot[IDIR]*3.14159/d.np_tot[JDIR]*KE << " " << std::endl;
  #endif
  f.close();
  }

}

// Initialisation routine. Can be used to allocate
// Arrays or variables which are used later on
Setup::Setup(Input &input, Grid &grid, DataBlock &data, Output &output) {
  // Get rotation rate along vertical axis
  omega=input.Get<real>("Hydro","rotation",0);
  shear=input.Get<real>("Hydro","shearingBox",0);
  psi=input.Get<real>("Hydro","warpMagnitude",0);
  cs=input.Get<real>("Hydro","csiso",1);


  tauGlob = input.Get<real>("Dust","drag",1);
  epsilon = input.Get<real>("Setup","epsilon",0);
  chi = input.Get<real>("Setup","chi",0);
  //data.hydro->EnrollUserSourceTerm(&PressureGradient);
  //data.hydro->EnrollUserSourceTerm(&InducedWarpFlow);

  // Add our userstep to the timeintegrator
  data.gravity->EnrollBodyForce(BodyForce);
  output.EnrollAnalysis(&Analysis);
  //data.hydro->EnrollUserDefBoundary(&reversePeriodicGas);
  //data.dust[0]->EnrollUserDefBoundary(&reversePeriodicDust);
  //data.hydro->EnrollUserDefBoundary(nonReflective);
  //data.dust[0]->EnrollUserDefBoundary(nonReflective);



}

// This routine initialize the flow
// Note that data is on the device.
// One can therefore define locally
// a datahost and sync it, if needed
void Setup::InitFlow(DataBlock &data) {
    // Create a host copy
    DataBlockHost d(data);

    real taus = tauGlob*omega;
    real D = 1+chi;
    real Lz = 1.0;
    real Kb = 2.*pi/Lz;
    real Kx = 15.6;
    real w_n1 = -0.373*omega;
    real w_n2 = 0.627*omega;
    real pert = 0.0001;
    real norm_n1 = pert/(w_n1*(w_n1*w_n1-Kb*Kb*1));
    real norm_n2 = pert/(w_n2*(w_n2*w_n2-Kb*Kb*4));

    for(int k = 0; k < d.np_tot[KDIR] ; k++) {
        for(int j = 0; j < d.np_tot[JDIR] ; j++) {
            for(int i = 0; i < d.np_tot[IDIR] ; i++) {
                real x=d.x[IDIR](i);
                real z=d.x[KDIR](k);

                d.Vc(RHO,k,j,i) = 1.0;
                d.Vc(RHO,k,j,i) *= exp(-norm_n1*Kx*w_n1*w_n1*sin(Kx*x+Kb*1*z));
                d.Vc(RHO,k,j,i) *= exp(-norm_n2*Kx*w_n2*w_n2*sin(Kx*x+Kb*2*z));

                d.Vc(VX1,k,j,i) = psi*cs*sin(Kb*z);
                d.Vc(VX1,k,j,i) += -pert*sin(Kx*x+Kb*1*z);
                d.Vc(VX1,k,j,i) += -pert*sin(Kx*x+Kb*2*z);
    
                d.Vc(VX2,k,j,i) = shear*x;
                d.Vc(VX2,k,j,i) += norm_n1*0.5*(w_n1*w_n1-Kb*Kb*1)*cos(Kx*x+Kb*1*z);
                d.Vc(VX2,k,j,i) += norm_n2*0.5*(w_n2*w_n2-Kb*Kb*4)*cos(Kx*x+Kb*2*z);

                d.Vc(VX3,k,j,i) = 0.0;
                d.Vc(VX3,k,j,i) += -norm_n1*Kx*Kb*1*w_n1*sin(Kx*x+Kb*1*z);
                d.Vc(VX3,k,j,i) += -norm_n2*Kx*Kb*2*w_n2*sin(Kx*x+Kb*2*z);

                d.dustVc[0](RHO,k,j,i) = chi;
                d.dustVc[0](RHO,k,j,i) *= exp(-0.00419650*cos(Kx*x+Kb*1*z)+0.00000823*sin(Kx*x+Kb*1*z));
                d.dustVc[0](RHO,k,j,i) *= exp(0.00256206*cos(Kx*x+Kb*2*z)-0.00000894*sin(Kx*x+Kb*2*z));

                d.dustVc[0](VX1,k,j,i) = psi*cs*sin(Kb*z);
                d.dustVc[0](VX1,k,j,i) += 0.00010000*cos(Kx*x+Kb*1*z);
                d.dustVc[0](VX1,k,j,i) += 0.00010000*cos(Kx*x+Kb*2*z);

                d.dustVc[0](VX2,k,j,i) = shear*x;
                d.dustVc[0](VX2,k,j,i) += 0.00000331*cos(Kx*x+Kb*1*z)-0.00003285*sin(Kx*x+Kb*1*z);
                d.dustVc[0](VX2,k,j,i) += 0.00000156*cos(Kx*x+Kb*2*z)+0.00004508*sin(Kx*x+Kb*2*z);

                d.dustVc[0](VX3,k,j,i) = 0.0;
                d.dustVc[0](VX3,k,j,i) += -0.00000098*cos(Kx*x+Kb*1*z)-0.00000363*sin(Kx*x+Kb*1*z);
                d.dustVc[0](VX3,k,j,i) += 0.00000649*cos(Kx*x+Kb*2*z)+0.00002271*sin(Kx*x+Kb*2*z);

            }
        }
    }

    // Send it all, if needed
    d.SyncToDevice();
}


// Analyse data to produce an output

void MakeAnalysis(DataBlock & data) {
}
