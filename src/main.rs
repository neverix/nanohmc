#![feature(trait_alias)]
use nannou::prelude::*;

fn main() {
    nannou::app(model)
        .update(update)
        .simple_window(view)
        .run();
}

#[derive(Default, Debug, Clone, Copy)]
struct PotentialGrid;

impl PotentialEnergy<Vec2> for PotentialGrid {
    fn energy(&self, position: Vec2) -> f32 {
        -(position - Vec2::new(0.5, 0.5)).length_squared()
    }

    fn gradient(&self, position: Vec2) -> Vec2 {
        (Vec2::new(0.5, 0.5) - position) * 2.0
    }
}

trait PotentialEnergy<T> {
    fn energy(&self, position: T) -> f32;
    fn gradient(&self, position: T) -> T;
}

trait AlmostFloat = Into<Vec<f32>> + From<Vec<f32>> + Clone;

#[derive(Debug, Clone, Copy)]
struct HMCConfig {
    step_size: f32,
    num_steps: usize,
    mass: f32,
}

#[derive(Debug)]
struct HMCState {
    position: Vec<f32>,
    momentum: Vec<f32>,
    config: HMCConfig
}

struct Model {
    grid: PotentialGrid,
    hmc: HMCState,
}

fn model(_app: &App) -> Model {
    Model {
        grid: PotentialGrid::default(),
        hmc: HMCState {
            position: vec![0.2, 0.7],
            momentum: vec![0.9, -0.3],
            config: HMCConfig {
                step_size: 0.01,
                num_steps: 10,
                mass: 1.0,
            }
        }
    }
}

fn update(_app: &App, _model: &mut Model, _update: Update) {
    
}

fn view(_app: &App, _model: &Model, frame: Frame){
    frame.clear(PURPLE);
}