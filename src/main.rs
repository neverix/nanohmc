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

impl HMCState {
    fn step(&mut self, energy: &dyn PotentialEnergy<Vec<f32>>) {

    }
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

fn view(app: &App, model: &Model, frame: Frame){
    frame.clear(PURPLE);
    let draw = app.draw();
    let (w, h) = app.main_window().inner_size_points();
    let (w, h) = (w as u32, h as u32);
    println!("w: {}, h: {}", w, h);
    let mut img = nannou::image::DynamicImage::new_rgb8(w, h);
    let i_to_pos = |i: u32| {
        let i = i as u32;
        let (x, y) = (i % w, i / w);
        let (x, y) = (x as f32 / w as f32, y as f32 / h as f32);
        Vec2::new(x, y)
    };
    let energies = (0..w*h).map(i_to_pos).map(|x| model.grid.energy(x)).collect::<Vec<_>>();
    let min_energy = energies.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_energy = energies.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let min_color = (58f32, 75f32, 84f32);
    let max_color = (255f32, 255f32, 255f32);
    let img_mut = img.as_mut_rgb8().unwrap();
    img_mut.pixels_mut().into_iter().enumerate().for_each(|(i, pixel)| {
        let energy = model.grid.energy(i_to_pos(i as u32));
        let energy = (energy - min_energy) / (max_energy - min_energy);
        let color = (min_color.0 + energy * (max_color.0 - min_color.0),
                        min_color.1 + energy * (max_color.1 - min_color.1),
                        min_color.2 + energy * (max_color.2 - min_color.2));
        pixel[0] = color.0 as u8;
        pixel[1] = color.1 as u8;
        pixel[2] = color.2 as u8;
    });
    let texture = wgpu::Texture::from_image(app, &img);
    draw.texture(&texture);
    draw.to_frame(app, &frame).unwrap()
}