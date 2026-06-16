#lang racket
(define (pluck nodes)
  (let loop ([ns nodes] [i 0] [best #f] [bi -1])
    (cond [(null? ns) (if best (list best bi) (list))]
          [else (define x (car ns))
                (if (and (even? x) (or (not best) (< x best)))
                    (loop (cdr ns) (+ i 1) x i)
                    (loop (cdr ns) (+ i 1) best bi))])))
