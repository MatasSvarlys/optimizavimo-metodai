use core::f64;

fn main() {
    //a = 0; b =3
    //starting points
    let x0 = vec![0.0, 0.0];
    let x1 = vec![1.0, 1.0];
    let xm= vec![0.0, 0.3];

    //test to see that point [0, 0] has no meaning when calculating gradient descent
    let ld = linear_descent(&vec![0.0, 0.0]);
    println!("max point: ({}, {}); max V value: {}", ld[0], ld[1], target_function(&ld));

    //outputs
    print_answer(x0);
    print_answer(x1);
    print_answer(xm);

    //just to see if it works at all
    print_answer(vec![0.3, 0.2]);
    
}

fn print_answer(n: Vec<f64>) -> (){
    //first check if a box like this can exist at all
    if check_if_box_is_correct(n[0], n[1], calc_last_variable(n[0], n[1])){
        let gradient = target_function_gradient(&n);
        println!("input: [{}, {}]; 3rd pair: {}; volume with these sides: {}; direction vector at point=[{}, {}]", n[0], n[1], calc_last_variable(n[0], n[1]), target_function(&n), gradient[0], gradient[1]);
        
        //linear descent results
        let ld: Vec<f64> = linear_descent(&n);
        println!("-Linear descent result: max point: ({}, {}); max V value: {}", ld[0], ld[1], target_function(&ld));
    
        let sd: Vec<f64> = steepest_descent(&n);
        println!("-Steepest descent result: max point: ({}, {}); max V value: {}", sd[0], sd[1], target_function(&sd));
    }
}

//if box total surface area is one, given 2 pairs of side surface areas, calculate last one 
fn calc_last_variable(a: f64, b: f64) -> f64{
    1.0-a-b
}

//variables are actually 2ab, 2ac, 2bc
fn check_if_box_is_correct(s_a_b: f64, s_a_c: f64, s_b_c: f64) -> bool{
    let surface_area:f64 = s_a_b+s_a_c+s_b_c;
    if surface_area != 1.0 {
        println!("Surface area does not add to 1");
        return false;
    }
    // if s_a_b <= 0.0 || s_a_c <= 0.0 || s_b_c <= 0.0{
    //     println!("Invalid box");
    //     return false;
    // }
    true
}

//variables are actually 2ab, 2ac, 2bc
fn calc_volume_squared(s_a_b: f64, s_a_c: f64, s_b_c: f64) -> f64{
    (s_a_b*s_a_c*s_b_c) / 8.0 //2ab*2ac*2cb=8V^2
}   

//calc volume from its square
fn target_function(x:&Vec<f64>) -> f64{
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



fn linear_descent(starting_point: &Vec<f64>) -> Vec<f64>{
    let mut next_point: Vec<f64> = starting_point.clone();
    let mut gradient = target_function_gradient(starting_point);
    let mut gradient_norm = (gradient[0].powf(2.0)+gradient[1].powf(2.0)).sqrt();
    let mut step_count = 0;

    //constants
    let gama = 0.3;
    let gradient_tolerance = 10e-4;
    
    //rest of steps
    while gradient_norm > gradient_tolerance {
        //move towards the gradient cus we want to maximize the func
        next_point[0] = next_point[0] + gama*gradient[0];
        next_point[1] = next_point[1] + gama*gradient[1];

        //calc things for next loop
        gradient = target_function_gradient(&next_point);
        gradient_norm = (gradient[0].powf(2.0)+gradient[1].powf(2.0)).sqrt();
        step_count+=1;
    }

    println!("step count: {}", step_count);
    return next_point
}

fn steepest_descent(starting_point: &Vec<f64>) -> Vec<f64>{
    fn backtracking_line_search(x: &Vec<f64>, gradient: &Vec<f64>) -> f64 {
        let alpha = 0.5;  // Step size reduction factor
        let beta = 0.8;   // Condition factor for sufficient decrease
        let mut step_size = 0.5;

        // Initial function value at current point
        let current_value = target_function(x);

        // Backtracking to find the optimal step size
        while step_size > 1e-8 {  // Prevent the step size from becoming too small
            let new_point = vec![
                x[0] + step_size * gradient[0],
                x[1] + step_size * gradient[1],
            ];

            let new_value = target_function(&new_point);

            // Check if this step size satisfies the condition
            if new_value > current_value + beta * step_size * (gradient[0].powi(2) + gradient[1].powi(2)) {
                return step_size;
            }

            // If not, reduce the step size
            step_size *= alpha;
        }

        step_size
    }

    let mut step_count = 0;
    let mut next_point = starting_point.clone();
    let mut gradient = target_function_gradient(starting_point);
    let mut gradient_norm = (gradient[0].powf(2.0)+gradient[1].powf(2.0)).sqrt();

    //constants
    let gradient_tolerance = 10e-4;
    
    while gradient_norm > gradient_tolerance {
        let optimal_step = backtracking_line_search(&next_point, &gradient);
        
        if optimal_step < 1e-8 {
            println!("Optimal step size too small, stopping...");
            break;
        }

        // println!("Optimal step size: {}", optimal_step);
        next_point[0]=next_point[0]+optimal_step*gradient[0];
        next_point[1]=next_point[1]+optimal_step*gradient[1];
        
        gradient = target_function_gradient(&next_point);
        gradient_norm = (gradient[0].powf(2.0)+gradient[1].powf(2.0)).sqrt();
        
        step_count+=1;
    }
    println!("step count: {}", step_count);
    next_point
}