#![feature(trait_alias)]
use nannou::{image::DynamicImage, prelude::*};
use nannou::rand::{self, Rng};
use rand_distr::Distribution;

fn main() {
    nannou::app(model)
        .update(update)
        .simple_window(view)
        .run();
}

trait TryToVec2 {
    fn try_to_vec2(&self) -> Option<Vec2>;
}

impl TryToVec2 for Vec<f32> {
    fn try_to_vec2(&self) -> Option<Vec2> {
        if self.len() == 2 {
            Some(Vec2::new(self[0], self[1]))
        } else {
            None
        }
    }
}

trait ToVec {
    fn to_vec(&self) -> Vec<f32>;
}

impl ToVec for Vec2 {
    fn to_vec(&self) -> Vec<f32> {
        vec![self.x, self.y]
    }
}


#[derive(Debug, Clone)]
struct PotentialGrid {
    peaks: Vec<Vec2>,
    scale: f32
}

impl Default for PotentialGrid {
    fn default() -> Self {
        PotentialGrid {
            peaks: vec![Vec2::new(0.1, 0.9), Vec2::new(0.4, 0.5), Vec2::new(0.6, 0.3)],
            scale: 4.0
        }
    }
}

impl PotentialEnergy<Vec2> for PotentialGrid {
    fn energy(&self, position: Vec2) -> f32 {
        let position = position * 2. - Vec2::ONE;
        let position = position * self.scale;
        self.peaks.iter().map(|&peak| (position - peak).length_squared()).sum()
    }

    fn gradient(&self, position: Vec2) -> Vec2 {
        let position = position * 2. - Vec2::ONE;
        let position = position * self.scale;
        self.peaks.iter().map(|&peak| (position - peak)).fold(Vec2::ZERO, std::ops::Add::add) * 2.0
    }
}

trait PotentialEnergy<T> {
    fn energy(&self, position: T) -> f32;
    fn gradient(&self, position: T) -> T;
}

impl PotentialEnergy<Vec<f32>> for PotentialGrid {
    fn energy(&self, position: Vec<f32>) -> f32 {
        self.energy(position.try_to_vec2().unwrap())
    }

    fn gradient(&self, position: Vec<f32>) -> Vec<f32> {
        self.gradient(position.try_to_vec2().unwrap()).to_vec()
    }
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
    config: HMCConfig,
    past_state: Option<(Vec<f32>, f32)>, // (position, energy)
    since_past: usize,  // number of steps since last past state
}

impl Default for HMCState {
    fn default() -> Self {
        return HMCState {
            position: vec![],
            momentum: vec![],
            config: HMCConfig {
                step_size: 1e-3,
                num_steps: 10,
                mass: 1.0,
            },
            past_state: None,
            since_past: 0,
        }
    }
}

impl HMCState {
    fn regenerate_momentum(&mut self) {
        let mut rng = rand::thread_rng();
        let normal_distribution = rand_distr::Normal::new(0.0, 1.0).unwrap();
        self.momentum = self.momentum.iter().map(|_| normal_distribution.sample(&mut rng)).collect();
    }

    fn step(&mut self, energy: &dyn PotentialEnergy<Vec<f32>>) {
        // leapfrog step
        // position update
        self.position = self.position.iter().zip(self.momentum.iter()).map(
                |(x, p)| x + self.config.step_size * p / self.config.mass
            ).collect();
        // momentum update
        let grad = energy.gradient(self.position.clone());
        self.momentum = self.momentum.iter().zip(grad.iter()).map(
                |(p,g)| p - self.config.step_size * g
            ).collect();
        self.since_past += 1;
        if self.since_past >= self.config.num_steps {
            let hamiltonian = energy.energy(self.position.clone())
                + 0.5 * self.momentum.iter().map(|p| p*p).sum::<f32>() * self.config.mass;
            let mut rng = rand::thread_rng();
            match &self.past_state {
                Some((past_position, past_hamiltonian)) => {
                    if hamiltonian < *past_hamiltonian {
                        self.position = past_position.clone();
                    } else {
                        let delta_hamiltonian = hamiltonian - past_hamiltonian;
                        let accept_prob = (delta_hamiltonian).exp();
                        if rng.gen::<f32>() < accept_prob {
                            self.past_state = Some((self.position.clone(), hamiltonian));
                        } else {
                            self.position = past_position.clone();
                        }
                    }
                },
                None => {
                    self.past_state = Some((self.position.clone(), hamiltonian));
                }
            }
            self.past_state = Some((self.position.clone(), hamiltonian));
            self.since_past = 0;
            self.regenerate_momentum();
        }
    }
}

struct Model {
    grid: PotentialGrid,
    hmc: HMCState,
    grid_texture: wgpu::Texture,
}

fn model(app: &App) -> Model {
    let img = DynamicImage::new_rgb8(2, 2);
    Model {
        grid: PotentialGrid::default(),
        hmc: HMCState {
            position: vec![0.2, 0.7],
            momentum: vec![0.9, -0.3],
            config: HMCConfig {
                step_size: 5e-3,
                num_steps: 40,
                mass: 1.0,
            },
            ..Default::default()
        },
        grid_texture: wgpu::Texture::from_image(app, &img)
    }
}

fn update(app: &App, model: &mut Model, _update: Update) {
    let (w, h) = app.main_window().inner_size_points();
    let (w, h) = (w.max(2.0) as u32, h.max(2.0) as u32);
    if [w, h] != model.grid_texture.size() {
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
        model.grid_texture = texture;
    }

    model.hmc.step(&model.grid);
}

fn view(app: &App, model: &Model, frame: Frame) {
    frame.clear(PURPLE);
    let draw = app.draw();
    draw.texture(&model.grid_texture);
    let (w, h) = app.main_window().inner_size_points();
    let ball = (&model.hmc.position, RED);
    for (position, color) in match &model.hmc.past_state {
        Some((position, _)) => vec![(position, YELLOW), ball],
        None => vec![ball],
    } {
        let particle = position.try_to_vec2().unwrap();
        let particle = particle - Vec2::new(0.5, 0.5);
        draw.xy(Vec2::new(w as f32, -h as f32) * particle).ellipse().radius(10f32).color(color);
    }
    draw.to_frame(app, &frame).unwrap();
}