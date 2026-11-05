use std::time::Instant;
fn main(){const W:usize=100;const WP:usize=102;
 let mut g=vec![0u8;W*W]; for i in 0..W*W { if (i*2654435761usize)%20==0 {g[i]=1;} }
 let table=[[0u8,0,1,1,1,1,1,1,1],[1;9]]; let mut nxt=g.clone(); let mut m=vec![0u8;WP*WP];
 let k=20000; let t=Instant::now();
 for _ in 0..k { for y in 0..W { for x in 0..W { m[(y+1)*WP+x+1]=g[y*W+x]; } }
  for y in 0..W { for x in 0..W { let c=(y+1)*WP+x+1;
   let n=m[c-WP-1]+m[c-WP]+m[c-WP+1]+m[c-1]+m[c+1]+m[c+WP-1]+m[c+WP]+m[c+WP+1];
   nxt[y*W+x]=table[g[y*W+x] as usize][n as usize]; } }
  std::mem::swap(&mut g,&mut nxt);}
 println!("rust {:.2} us/step (checksum {})", t.elapsed().as_secs_f64()/k as f64*1e6, g.iter().map(|&v| v as u32).sum::<u32>());}
