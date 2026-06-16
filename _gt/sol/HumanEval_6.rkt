#lang racket
(define (parse_nested_parens s)
  (map (lambda (grp)
         (let loop ([chars (string->list grp)] [depth 0] [mx 0])
           (cond [(null? chars) mx]
                 [(char=? (car chars) #\() (loop (cdr chars) (+ depth 1) (max mx (+ depth 1)))]
                 [(char=? (car chars) #\)) (loop (cdr chars) (- depth 1) mx)]
                 [else (loop (cdr chars) depth mx)])))
       (string-split s)))
