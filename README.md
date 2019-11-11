# free_bandit
Simulation and experiment results of multi-armed bandits with free pulls.


## Motivation

We consider a smart home scheduler, i.e. a scheduler for Roomba, that tries to learn the user's true preferences while minimizing his regrets. It can either (1) deploy a schedule and get response from the user or (2) ask about the user's opinion on a schedule in a certain frequence. We map this setting to a multi-armed bandit problem with free pulls. The difference from a normal MAB problem is the bandit can have a "free pull" every k rounds, which receives the rewards signal but incurs no real regrets. 

The settings is: 

We have a finite set of arms, corresponding to a finite set of daily schedules. Because an event can start at any time, meaning there're continuously many arms, we apply lipschtiz assumptions and discretization to get a reasonable finite set of arms.

The reward is a Bernouli random variable, 1 for the user not changing the schedule, 0 for changing. An important assumption is if the user changes the schedule, the changed schedule doesn't tell the best arm. And we don't consider the comparison information that the schdule after is better than before, which we may consider in future analysis. It means we only care the fact that the user changes the schedule, but ignore what schedule it is changed to.

A free pull means the scheduler can ask the user if he will change a schedule. It doesn't incur the same regrets as normal pulls because the scheduler hasn't been deployed and we consider the cost of asking is trivial. In fact the cost can be seemed measured as the frequence. Larger cost means less frequence and vice versa. 


### Other possibly applicable scenarios(other discussed settings)
Online advertisements. Problems: (1) there's no proper free pulls. Rewards of asking questions are quite different from click rates. (2) There's no reason free pulls can't be used up at the beginning.

Consulting. Consider consulting companies' research as free pulls. Problems: (1) the time horizon T might be too small if we consider the research frequency. For example, the consulting company reports every month. (2) the true rewards will change in such long time. (3) there should be an option of not using such "free" pulls - because the cost of consulting is not trivial.

A scheduler for blackouts. Free pulls are asking citizens if a schedule is good. Problems: it's more propriate that the blackouts should be scheduled according to domain knowledge. For example, instead of scheduling an outage at certain time, a place in danger will be blackouted for the whole time until the danger is lowered to a certain level.


### Future analysis:

* Comparison information.
* Strong/weak pull
* Can choose not to have a free pull
* contextual bandits