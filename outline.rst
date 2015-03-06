Authors
=======

- Jason K. Moore
- Antonie van den Bogert

Introduction
============

Things to cite
--------------

- Manoj's recent paper that got a "control" law.
- Wang's predictive simulation
- Thomas Geitenbeek predictive simulation
- Elliot Rouse's work on the ankle.
- van der Kooij paper on direct id
- Probably some Kearney on id
- Neural network for bicylce tricks from Georgia Tech
- Hof's work
- GEYER, H., AND HERR, H. 2010. A muscle-reflex model that en-
  codes principles of legged mechanics produces human walking
  dynamics and muscle activities

- Note the need for high enough perturbations to have any change of getting
  accurate gain estimates from the controller.
- Talk about the control identification problem

Methods
-------

- Experiments: Basically a citation to the data paper but describe which
  trials we focus on and why.
- Give some plots showing the perturbed vs unperturbed for the trials we had
  (like the data paper).
- Describe the closed loop system architecture.
- Desribe the controller: gait phase scheduling, non-linear
- Describe direct identification
- Describe the linear least squares that provides the results
- Describe how we validate the model on independent data
- Show that artificially created variations in the data do not produce the same
  results.
- Show what happens if only m* is identified versus all gains.
- Give all the input values for the data preparation: filter freq, grf
  thresholds
- Describe the 2D inverse dynamics
- Describe the data cleaning

Results
-------

- Example gains for one trial
- Mean gains for all trials
- Show some validation results, i.e. how well the model predictions fit
  independent data.
- Show that it doesn't matter if we compensate the forces/moments wrt belt
  acceleration.

Discussion
----------

- How do we know if we've perturbed enough?
- What does it mean when we get similar results from perturbed and unperturbed
  data.
- Are the forces good quality without compensation?
- What are the negative gains all about?
- Can we connect this to the physiology?
- Do we really have a controller description here?
- How can this be useful?
