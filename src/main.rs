use core::f64;

fn main() {
    //a = 0; b =3
    //starting points
    let x0 = vec![0.0, 0.0];
    let x1 = vec![1.0, 1.0];
    let xm= vec![0.0, 0.3];

    //outputs
    print_answer(x0);
    print_answer(x1);
    print_answer(xm);
}

fn print_answer(n: Vec<f64>) -> (){
    //first check if a box like this can exist at all
    // if check_if_box_is_correct(n[0], n[1], calc_last_variable(n[0], n[1])){
        let gradient = target_function_gradient(&n);
        println!("\n\nrecieved point: {:?} (3rd var = {}); volume with these sides: {}; gradient at point={:?}", n, 1.0-n[0]-n[1], target_function(&n), gradient);
        if gradient == vec![0.0, 0.0]{
            println!("\ngradient at point is 0, cannot use descent methods");
        } else {
            //linear descent results
            println!("\nstarting linear descent");
            let ld: Vec<f64> = linear_descent(&n);
            println!("-Linear descent result: max point: ({}, {}); max V value: {}", ld[0], ld[1], target_function(&ld));
            
            //steepest descent
            println!("\nstarting steepest descent");
            let sd: Vec<f64> = steepest_descent(&n);
            println!("-Steepest descent result: max point: ({}, {}); max V value: {}", sd[0], sd[1], target_function(&sd));
        }

        //simplex method
        println!("\nstarting simplex reduction");
        let simplex_answ = simplex(&n);
        if !target_function(&simplex_answ).is_nan(){
            println!("-Simplex method result: found (x, y): {:?}, value at found point: {}", simplex_answ.clone(), target_function(&simplex_answ));
        }
    // }
}

//calc volume from its square
fn target_function(x:&Vec<f64>) -> f64{
    //if box total surface area is one, given 2 pairs of side surface areas, calculate last one 
    fn calc_last_variable(a: f64, b: f64) -> f64{
        1.0-a-b
    }
    //variables are actually 2ab, 2ac, 2bc
    fn calc_volume_squared(ab: f64, ac: f64, bc: f64) -> f64{
        (ab*ac*bc) / 8.0 //2ab*2ac*2cb=8V^2
    }   

    calc_volume_squared(x[0], x[1], calc_last_variable(x[0], x[1])).sqrt()
}

//derivatives by derivative calculator, idfk how to do multivariable differenciation
fn target_function_gradient(x:&Vec<f64>) -> Vec<f64>{
    fn dx(x: f64, y: f64) -> f64{
        y*(-2.0*x-y+1.0)
    }

    fn dy(x: f64, y: f64) -> f64{
        x*(1.0-x-2.0*y)
    }

    vec![dx(x[0], x[1]), dy(x[0], x[1])]
}
fn calc_gradient_norm(gradient_vec: &Vec<f64>) -> f64{
    (gradient_vec[0].powf(2.0)+gradient_vec[1].powf(2.0)).sqrt()
}


fn linear_descent(starting_point: &Vec<f64>) -> Vec<f64>{
    let mut curr_point = starting_point.clone();
    let mut gradient = target_function_gradient(starting_point);
    let mut gradient_norm = calc_gradient_norm(&gradient);
    let mut step_count = 1;
    
    //constants
    let gama = 0.3; //step size
    let gradient_tolerance = 10e-4;
    
    //rest of steps
    while gradient_norm > gradient_tolerance {
        println!("{}. Currently at point: {:?}, with f(x, y) = {}", step_count, curr_point, target_function(&curr_point));
        println!("    Gradient at step {}: {:?}", step_count, gradient);
        //move towards the gradient cus we want to maximize the func
        let next_point: Vec<f64> = vec![curr_point[0] + gama*gradient[0], curr_point[1] + gama*gradient[1]];

        //calc things for next loop
        gradient = target_function_gradient(&next_point);
        gradient_norm = calc_gradient_norm(&gradient);
        
        curr_point = next_point;
        step_count+=1;
    }

    println!("step count: {}", step_count);
    return curr_point
}

fn steepest_descent(starting_point: &Vec<f64>) -> Vec<f64>{
    fn optimize_step(x: &Vec<f64>, gradient: &Vec<f64>) -> f64 {
        let mut step_size = 0.01; //linear descents gama
        let mut potential_step_size = step_size;

        let gama = 1.2; //ammount to multiply the step size by to get potential step size for next loop

        // Initial function value at current point
        let mut current_value = target_function(x);

        //take a step in the direction of the gradient
        let mut new_point = vec![
            x[0] + step_size * gradient[0],
            x[1] + step_size * gradient[1],
        ];

        //while the new point increaces the value of the output, keep stepping in that direction
        while target_function(&new_point) > current_value || current_value.is_nan() {
            //if the potential step size passed the check, make it the current step size
            step_size = potential_step_size;

            //update the current best function value
            current_value = target_function(&new_point);

            //potentially update step size
            potential_step_size = gama*step_size;

            //step in specified direction
            new_point = vec![
                x[0] + potential_step_size * gradient[0],
                x[1] + potential_step_size * gradient[1],
            ];
        }

        step_size
    }

    //init variables
    let mut step_count = 1;
    let mut next_point = starting_point.clone();
    let mut gradient = target_function_gradient(starting_point);
    let mut gradient_norm = calc_gradient_norm(&gradient);

    //constants
    let gradient_tolerance = 10e-4;
    
    while gradient_norm > gradient_tolerance {
        
        //choose an optimal step size with the algorythm above
        let optimal_step = optimize_step(&next_point, &gradient);

        println!("{}. Currently at point: {:?}, with f(x, y) = {}; Optimized step size for next step: {}", step_count, next_point, target_function(&next_point), optimal_step);
        println!("    Gradient at step {}: {:?}", step_count, gradient);

        //step in gradients direction since we're maximizing -f(x, y)
        next_point[0]=next_point[0]+optimal_step*gradient[0];
        next_point[1]=next_point[1]+optimal_step*gradient[1];
        
        //recalc vars for next loop
        gradient = target_function_gradient(&next_point);
        gradient_norm = (gradient[0].powf(2.0)+gradient[1].powf(2.0)).sqrt();
        
        step_count+=1;

    }
    println!("step count: {}", step_count-1);
    next_point
}

fn simplex(starting_point: &Vec<f64>) -> Vec<f64> {
    //helper funcs
    fn clamp(value: f64, min: f64, max: f64) -> f64 {
        if value < min {
            min
        } else if value > max {
            max
        } else {
            value
        }
    }
    fn create_initial_simplex(starting_point: &Vec<f64>, side_length: f64) -> Vec<Vec<f64>> {
        let sqrt_3 = f64::sqrt(3.0);
        let n0 = vec![
            clamp(starting_point[0], 0.0, 1.0),
            clamp(starting_point[1] + sqrt_3 / 3.0 * side_length, 0.0, 1.0),
        ];
        let n1 = vec![
            clamp(starting_point[0] + side_length / 2.0, 0.0, 1.0),
            clamp(starting_point[1] - sqrt_3 / 6.0 * side_length, 0.0, 1.0),
        ];
        let n2 = vec![
            clamp(starting_point[0] - side_length / 2.0, 0.0, 1.0),
            clamp(starting_point[1] - sqrt_3 / 6.0 * side_length, 0.0, 1.0),
        ];

        let ret = order_simplex(vec![n0, n1, n2]);
        ret
    }
    fn order_simplex(simplex: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        let mut points = vec![
            (simplex[0].clone(), target_function(&simplex[0])),
            (simplex[1].clone(), target_function(&simplex[1])),
            (simplex[2].clone(), target_function(&simplex[2])),
        ];
        //sort points by function value in descending order
        points.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        //return only the sorted points
        points.into_iter().map(|(point, _)| point).collect()
    }
    fn calc_mid_point(point_a: &Vec<f64>, point_b: &Vec<f64>) -> Vec<f64> {
        vec![(point_a[0] + point_b[0]) / 2.0, (point_a[1] + point_b[1]) / 2.0]
    }
    fn calc_distance_between(point_a: &Vec<f64>, point_b: &Vec<f64>) -> f64 {
        ((point_a[0] - point_b[0]).powi(2) + (point_a[1] - point_b[1]).powi(2)).sqrt()
    }
    fn calc_reflected_point(worst_point: Vec<f64>, mid_point: Vec<f64>) -> Vec<f64> {
        vec![2.0 * mid_point[0] - worst_point[0], 2.0 * mid_point[1] - worst_point[1]]
    }
    fn shrink_simplex(ordered_simplex: &mut Vec<Vec<f64>>) {
        ordered_simplex[0] = calc_mid_point(&ordered_simplex[0], &ordered_simplex[2]);
        ordered_simplex[1] = calc_mid_point(&ordered_simplex[1], &ordered_simplex[2]);
    }

    //main loop
    let initial_side_length = 0.5;
    let min_side_length = 10e-4;
    let mut step_count = 1;
    //create a simplex
    let mut simplex = create_initial_simplex(starting_point, initial_side_length);
    if target_function(&simplex[1]).is_nan() && target_function(&simplex[0]).is_nan() && target_function(&simplex[2]).is_nan(){
        println!("Could not find function values at any of the simplex corners");
        return simplex[0].clone();
    }
    //while any of the side lengths are big enough
    while calc_distance_between(&simplex[0], &simplex[1]) > min_side_length || 
          calc_distance_between(&simplex[0], &simplex[2]) > min_side_length || 
          calc_distance_between(&simplex[1], &simplex[2]) > min_side_length {

        //find the reflected point from the worst point (i=2 in array) and the midpoint between the left over 2 
        let current_reflected_point = calc_reflected_point(simplex[2].clone(), calc_mid_point(&simplex[0], &simplex[1]));
        println!("{}. Simplex: [best:{:?}, mid:{:?}, worst:{:?}], reflected point: {:?}, f(x,y) at best point: {}", step_count, simplex[0], simplex[1], simplex[2], current_reflected_point, target_function(&simplex[0]));

        //check if we moved in the correct direction at least enough to get a better point
        if target_function(&current_reflected_point) > target_function(&simplex[1]){
            //replace worst point and sort simplex points by function values
            println!("    Replaced worst point");
            simplex[2] = current_reflected_point.clone();
            simplex = order_simplex(simplex);
        } else {
            //shrink towards best point
            println!("    Shrunk simplex");
            shrink_simplex(&mut simplex);
            simplex = order_simplex(simplex);
        }
        step_count+=1;
    }

    simplex[0].clone() // This returns the point with the maximum function value
}
