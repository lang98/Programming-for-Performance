use super::checksum::Checksum;
use super::Event;
use crossbeam::channel::Sender;
use std::sync::{Arc, Mutex};

pub struct Idea {
    pub name: String,
    pub num_pkg_required: usize,
}

pub struct IdeaGenerator {
    products: Vec<String>,
    customers: Vec<String>,
    idea_start_idx: usize,
    num_ideas: usize,
    num_students: usize,
    num_pkgs: usize,
    event_sender_new: Sender<Event>,
    event_sender_out: Sender<Event>,
}

impl IdeaGenerator {
    pub fn new(
        product_file: String,
        customer_file: String, 
        idea_start_idx: usize,
        num_ideas: usize,
        num_students: usize,
        num_pkgs: usize,
        event_sender_new: Sender<Event>,
        event_sender_out: Sender<Event>,
    ) -> Self {
        Self {
            customers: customer_file.lines().map(|x| x.into()).collect(),
            products: product_file.lines().map(|x| x.into()).collect(),
            idea_start_idx,
            num_ideas,
            num_students,
            num_pkgs,
            event_sender_new,
            event_sender_out,
        }
    }

    // Idea names are generated from cross products between product names and customer names
    fn get_next_idea_name(&self, idx: usize) -> String {
        let ideas = Self::cross_product(self.products.clone(), self.customers.clone());
        let pair = &ideas[idx % ideas.len()];
        format!("{} for {}", pair.0, pair.1)
    }

    fn cross_product(products: Vec<String>, customers: Vec<String>) -> Vec<(String, String)> {
        products
            .iter()
            .flat_map(|p| customers.iter().map(move |c| (p.to_owned(), c.to_owned())))
            .collect()
    }

    pub fn run(&self, idea_checksum: Arc<Mutex<Checksum>>) {
        let pkg_per_idea = self.num_pkgs / self.num_ideas;
        let extra_pkgs = self.num_pkgs % self.num_ideas;

        let mut checksum = Checksum::default();

        // Generate a set of new ideas and place them into the event-queue
        // Update the idea checksum with all generated idea names
        for i in 0..self.num_ideas {
            let name = self.get_next_idea_name(self.idea_start_idx + i);
            let extra = (i < extra_pkgs) as usize;
            let num_pkg_required = pkg_per_idea + extra;
            let idea = Idea {
                name,
                num_pkg_required,
            };

            checksum.update(Checksum::with_sha256(&idea.name));

            self.event_sender_new.send(Event::NewIdea(idea)).unwrap();
        }
        idea_checksum
            .lock()
            .unwrap()
            .update(checksum);

        // Push student termination events into the event queue
        for _ in 0..self.num_students {
            self.event_sender_out.send(Event::OutOfIdeas).unwrap();
        }
    }
}
