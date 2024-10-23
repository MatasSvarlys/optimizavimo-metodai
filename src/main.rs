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
    //helper funcs
    fn calc_surface_areas(x: f64, y: f64, z: f64) -> Vec<f64>{
        vec![2.0*x*y, 2.0*y*z, 2.0*x*z]
    }


    let s_x = calc_surface_areas(n[0], n[1], calc_last_variable(n[0], n[1]));
    //first check if a box like this can exist at all
    if check_if_box_is_correct(s_x[0], s_x[1], s_x[2]){
        let gradient = target_function_gradient(&n);
        println!("input: [{}, {}]; last var: {}; f(X)={}; delta f(X)=[{}, {}]", n[0], n[1], calc_last_variable(n[0], n[1]), target_function(&n), gradient[0], gradient[1]);
        
        //linear descent results
        let ld: Vec<f64> = linear_descent(&n);
        println!("max point: ({}, {}); max V value: {}", ld[0], ld[1], target_function(&ld));
    
    }
}

//if box surface area is one, given 2 side lenghts of the box, calculate last one 
fn calc_last_variable(a: f64, b: f64) -> f64{
    (0.5-a*b)/(a+b)
}

//variables are actually 2ab, 2ac, 2bc
fn check_if_box_is_correct(s_a_b: f64, s_a_c: f64, s_b_c: f64) -> bool{
    let surface_area:f64 = s_a_b+s_a_c+s_b_c;
    if surface_area != 1.0 {
        println!("Surface area does not add to 1");
        return false;
    }
    if s_a_b <= 0.0 || s_a_c <= 0.0 || s_b_c <= 0.0{
        println!("Invalid box");
        return false;
    }
    true
}

//variables are actually 2ab, 2ac, 2bc
//idfk where to use this, but in step 2 they ask to have a function for this :(((
fn calc_volume_squared(s_a_b: f64, s_a_c: f64, s_b_c: f64) -> f64{
    s_a_b+s_a_c+s_b_c / 8.0 //2ab*2ac*2cb=8V^2
}   

//just the volume of the box given all side lenghts
fn target_function(x:&Vec<f64>) -> f64{
    x[0]*x[1]*calc_last_variable(x[0], x[1])
}

//derivatives by derivative calculator, idfk how to do multivariable differenciation
fn target_function_gradient(x:&Vec<f64>) -> Vec<f64>{
    fn dx(x: f64, y: f64) -> f64{
        y*(-1.0*x.powf(2.0)*y-x*2.0*y.powf(2.0)+0.5*y)/(x+y).powf(2.0)
    }

    fn dy(x: f64, y: f64) -> f64{
        x*(-1.0*x*y.powf(2.0)-x.powf(2.0)*2.0*y+0.5*x)/(x+y).powf(2.0)
    }

    vec![dx(x[0], x[1]), dy(x[0], x[1])]
}



fn linear_descent(starting_point: &Vec<f64>) -> Vec<f64>{
    let mut next_point: Vec<f64> = starting_point.clone();
    let mut gradient = target_function_gradient(starting_point);
    let mut gradient_norm = (gradient[0].powf(2.0)+gradient[1].powf(2.0)).sqrt();
    let mut step_count = 0;

    //constants
    let gama = 0.08;
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

/*
fn steepest_descent(starting_point: &Vec<f64>) -> Vec<f64>{
    fn calc_optimal_step_size() -> f64{

    }
    
    let mut step_count = 0;


    //constants
    let gama = 0.08;
    let gradient_tolerance = 10e-4;
    
}
*/